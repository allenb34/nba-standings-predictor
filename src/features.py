"""Build the 4 features + target per (team, season) -> data/processed/features.csv.

Seasons are keyed by end year: 2012 = 2011-12. Training targets 2012..2026, prediction 2027 (2026-27).
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as static_teams

from fetch import RAW, ROOT, BBREF_YEARS, TEAM_SEASONS

PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

REPLACEMENT_BPM = -2.0   # pre-chosen, not tuned
CAP_MINUTES = 500.0      # weight cap for players with no t-1 row (pre-chosen, not tuned)
TRAIN_SEASONS = range(2012, 2027)
PREDICT_SEASON = 2027

BBREF_TO_NBA = {"BRK": "BKN", "NJN": "BKN", "CHO": "CHA", "CHA": "CHA", "PHO": "PHX", "NOH": "NOP"}
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}

# nba_api name -> BBRef slug, filled in by hand for veterans the name normalizer can't match.
MANUAL_IDS: dict[str, str] = {"Ronald Holland II": "hollaro01"}  # BBRef: "Ron Holland"


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def load_players() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (season totals per player, minutes per player-team-season)."""
    raw = pd.concat([pd.read_csv(RAW / f"bbref_advanced_{y}.csv") for y in BBREF_YEARS], ignore_index=True)
    raw["team"] = raw["team_bbref"].map(lambda t: BBREF_TO_NBA.get(t, t))
    raw = raw.sort_values(["season_end", "row_order"])

    team_rows = raw[raw["team"] != "TOT"]
    # TOT row if traded, else the single team row
    totals = (raw.assign(is_tot=raw["team"] == "TOT")
              .sort_values(["season_end", "player_id", "is_tot"], ascending=[True, True, False])
              .groupby(["season_end", "player_id"]).first()[["player", "age", "mp", "bpm"]])
    g = team_rows.groupby(["season_end", "player_id"])["team"]
    totals["first_team"] = g.first()   # rows are chronological (asserted below)
    totals["last_team"] = g.last()
    totals = totals.reset_index()

    per_team = team_rows.groupby(["season_end", "player_id", "team"], as_index=False)["mp"].sum()

    # Row order must be chronological, not by minutes: Doncic played fewer DAL than LAL minutes in 2024-25.
    ft = totals.set_index(["season_end", "player_id"])["first_team"]
    assert ft[(2025, "doncilu01")] == "DAL" and ft[(2022, "hardeja01")] == "BKN", "BBRef team rows not chronological"

    teams_per_season = per_team.groupby("season_end")["team"].nunique()
    assert (teams_per_season == 30).all(), teams_per_season[teams_per_season != 30]
    return totals, per_team


def load_teams() -> pd.DataFrame:
    abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    df = pd.concat([pd.read_csv(RAW / f"team_advanced_{s}.csv") for s in TEAM_SEASONS], ignore_index=True)
    df["team"] = df["TEAM_ID"].map(abbr)
    df["season_end"] = df["season"].str[:4].astype(int) + 1
    assert df["team"].notna().all() and (df.groupby("season_end")["team"].nunique() == 30).all()
    return df[["season_end", "team", "GP", "W", "W_PCT", "NET_RATING"]]


# Player-role weighting. Official settings are v1.1 (tag v1.1-lock); v1 (tag v1-lock) was
# {"weight": "t1", "bpm": "t1", "top_k": None}. See ROSTER_WEIGHTS_REVIEW.md and src/weight_variants.py.
#   weight="healthy": role = max(t-1, t-2 minutes); a t-2-only player keeps his full t-2 minutes (no 500 cap)
#   bpm="blend2":     minutes-weighted average of t-1 and t-2 BPM
#   top_k=10:         roster BPM and age use only the 10 heaviest-weighted players
VARIANT = {"weight": "healthy", "bpm": "blend2", "top_k": 10}


def player_inputs(ids: pd.Series, t: int, totals: pd.DataFrame) -> pd.DataFrame:
    """Weight (minutes) and BPM per player for target season t, using the t-1 / t-2 / replacement rule."""
    by = totals.set_index(["season_end", "player_id"])
    out = []
    for pid in ids:
        r2 = by.loc[(t - 2, pid)] if (t - 2, pid) in by.index else None
        if (t - 1, pid) in by.index:
            r = by.loc[(t - 1, pid)]
            w, bpm, src = r["mp"], r["bpm"], "t-1"
            if r2 is not None and VARIANT["weight"] == "healthy":
                w = max(w, r2["mp"])
            if r2 is not None and VARIANT["bpm"] == "blend2" and pd.notna(bpm) and pd.notna(r2["bpm"]):
                bpm = (r["mp"] * bpm + r2["mp"] * r2["bpm"]) / (r["mp"] + r2["mp"])
        elif r2 is not None:
            w = r2["mp"] if VARIANT["weight"] == "healthy" else min(r2["mp"], CAP_MINUTES)
            bpm, src = r2["bpm"], "t-2"
        else:
            w, bpm, src = CAP_MINUTES, REPLACEMENT_BPM, "replacement"
        if pd.isna(bpm):  # BBRef leaves BPM blank for tiny-minute players
            bpm = REPLACEMENT_BPM
        out.append({"player_id": pid, "w": w, "bpm": bpm, "src": src})
    return pd.DataFrame(out)


def team_features(roster: pd.DataFrame, t: int, totals: pd.DataFrame, per_team: pd.DataFrame) -> dict:
    """roster: one team's players for season t, columns player_id, team, age_t."""
    team = roster["team"].iloc[0]
    p = player_inputs(roster["player_id"], t, totals).merge(roster[["player_id", "age_t"]], on="player_id")
    if VARIANT["top_k"]:
        p = p.nlargest(VARIANT["top_k"], "w")
    prev = per_team[(per_team["season_end"] == t - 1) & (per_team["team"] == team)]
    return {
        "roster_bpm": np.average(p["bpm"], weights=p["w"]),
        "continuity": prev.loc[prev["player_id"].isin(roster["player_id"]), "mp"].sum() / prev["mp"].sum(),
        "roster_age": np.average(p["age_t"], weights=p["w"]),
        "n_players": len(p),
        "n_replacement": int((p["src"] == "replacement").sum()),
    }


def historical_rows(totals, per_team, teams) -> pd.DataFrame:
    rows = []
    for t in TRAIN_SEASONS:
        season = totals[totals["season_end"] == t].rename(columns={"first_team": "team", "age": "age_t"})
        for team, roster in season.groupby("team"):
            rows.append({"season_end": t, "team": team, **team_features(roster, t, totals, per_team)})
    return pd.DataFrame(rows)


def map_2026_27_roster(totals: pd.DataFrame) -> pd.DataFrame:
    ros = pd.read_csv(RAW / "rosters_2026-27.csv")
    assert not ros["PLAYER_ID"].duplicated().any(), "player on two 2026-27 rosters"
    recent = totals[totals["season_end"] >= PREDICT_SEASON - 2].sort_values("season_end")
    recent = recent.assign(key=recent["player"].map(norm_name))
    ids_per_key = recent.groupby("key")["player_id"].unique()
    ambiguous = ids_per_key[ids_per_key.map(len) > 1]
    if len(ambiguous):
        print("AMBIGUOUS name matches (resolve via MANUAL_IDS):", ambiguous.to_dict())
    key_to_id = recent.groupby("key")["player_id"].last()  # latest season wins

    ros["key"] = ros["PLAYER"].map(norm_name)
    ros["player_id"] = ros["PLAYER"].map(MANUAL_IDS).fillna(ros["key"].map(key_to_id))
    vets_unmatched = ros[ros["player_id"].isna() & (ros["EXP"].astype(str) != "R")]
    print(f"\n2026-27 roster: {len(ros)} players, {ros['player_id'].isna().sum()} unmatched "
          f"({len(vets_unmatched)} non-rookies)")
    if len(vets_unmatched):
        print(vets_unmatched[["team", "PLAYER", "EXP"]].to_string(index=False))
    rookies_matched = ros[ros["player_id"].notna() & (ros["EXP"].astype(str) == "R")]
    if len(rookies_matched):
        print("Rookies (EXP=R) that matched a BBRef id -- check these aren't name collisions:")
        print(rookies_matched[["team", "PLAYER", "player_id"]].to_string(index=False))

    # Unmatched players get a unique placeholder id so they fall through to replacement level.
    ros["player_id"] = ros["player_id"].fillna("nba_" + ros["PLAYER_ID"].astype(str))
    last = totals[totals["season_end"] == PREDICT_SEASON - 1].set_index("player_id")
    ros["age_t"] = ros["player_id"].map(last["age"] + 1).fillna(ros["AGE"])
    return ros


def spot_checks(ros: pd.DataFrame, totals: pd.DataFrame) -> None:
    last = totals[totals["season_end"] == PREDICT_SEASON - 1].set_index("player_id")
    inputs = player_inputs(ros["player_id"], PREDICT_SEASON, totals).set_index("player_id")
    view = ros.assign(
        team_2025_26=ros["player_id"].map(last["last_team"]),
        mp_2025_26=ros["player_id"].map(last["mp"]),
        bpm_2025_26=ros["player_id"].map(last["bpm"]),
        weight=ros["player_id"].map(inputs["w"]),
        bpm_used=ros["player_id"].map(inputs["bpm"]),
        source=ros["player_id"].map(inputs["src"]),
    )
    cols = ["team", "PLAYER", "player_id", "team_2025_26", "mp_2025_26", "bpm_2025_26", "weight", "bpm_used", "source"]
    for team in ["PHI", "MIA"]:
        print(f"\n=== {team} 2026-27 roster ===")
        print(view[view["team"] == team][cols].sort_values("weight", ascending=False).to_string(index=False))

    team_of = view.set_index("PLAYER")["team"]
    for name, want in [("LeBron James", "PHI"), ("Jaylen Brown", "PHI"), ("Giannis Antetokounmpo", "MIA")]:
        got = team_of.get(name)
        assert got == want, f"{name}: expected {want}, roster says {got}"

    changes = view[view["team_2025_26"].notna() & (view["team"] != view["team_2025_26"])]
    changes = changes[cols].sort_values("mp_2025_26", ascending=False)
    OUTPUTS.mkdir(exist_ok=True)
    changes.to_csv(OUTPUTS / "team_changes_2026_27.csv", index=False)
    print(f"\n=== {len(changes)} players whose 2026-27 team differs from their (final) 2025-26 team ===")
    print(changes[["PLAYER", "team_2025_26", "team", "mp_2025_26", "bpm_2025_26"]].to_string(index=False))


def build_features(totals, per_team, teams, ros) -> pd.DataFrame:
    hist = historical_rows(totals, per_team, teams)
    pred = pd.DataFrame([{"season_end": PREDICT_SEASON, "team": team,
                          **team_features(r, PREDICT_SEASON, totals, per_team)}
                         for team, r in ros.groupby("team")])

    df = pd.concat([hist, pred], ignore_index=True)
    cur = teams.rename(columns={"W_PCT": "win_pct", "GP": "gp"})[["season_end", "team", "win_pct", "gp"]]
    prev = teams.assign(season_end=teams["season_end"] + 1).rename(
        columns={"W_PCT": "win_pct_prev", "NET_RATING": "net_rtg_prev"})[["season_end", "team", "win_pct_prev", "net_rtg_prev"]]
    df = df.merge(cur, on=["season_end", "team"], how="left").merge(prev, on=["season_end", "team"], how="left")
    df["conf"] = np.where(df["team"].isin(EAST), "East", "West")

    train, future = df[df["season_end"] < PREDICT_SEASON], df[df["season_end"] == PREDICT_SEASON]
    assert len(train) == 450 and len(future) == 30, (len(train), len(future))
    feats = ["net_rtg_prev", "roster_bpm", "continuity", "roster_age", "win_pct_prev"]
    assert not train[feats + ["win_pct"]].isna().any().any(), train[train[feats + ["win_pct"]].isna().any(axis=1)]
    assert not future[feats].isna().any().any()
    assert (future["conf"].value_counts() == 15).all()
    return df


def main():
    totals, per_team = load_players()
    teams = load_teams()
    ros = map_2026_27_roster(totals)
    spot_checks(ros, totals)
    df = build_features(totals, per_team, teams, ros)
    train = df[df["season_end"] < PREDICT_SEASON]
    feats = ["net_rtg_prev", "roster_bpm", "continuity", "roster_age", "win_pct_prev"]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "features.csv", index=False)
    print(f"\nwrote features.csv: {len(train)} train rows, {(df["season_end"] == PREDICT_SEASON).sum()} 2026-27 rows")
    print(train[feats + ["win_pct"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
