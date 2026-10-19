"""Weekly tracker: current 2026-27 standings vs the LOCKED predictions -> docs/data.json.

Reads outputs/predictions_2026_27.csv (never writes it). The raw standings pull is cached per day at
data/raw/standings/standings_2026-27_{YYYY-MM-DD}.csv, so rerunning the same day makes zero network calls.
Rerun weekly through the end of the regular season (2027-04-11):  python src/update_actuals.py
"""
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from fetch import _retry  # also injects truststore before any HTTPS
from nba_api.stats.endpoints import leaguestandingsv3
from nba_api.stats.static import teams as static_teams

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "outputs" / "predictions_2026_27.csv"
# Exploratory coach-adjusted v2 (branch v2-coach-effects), tracked alongside but never replacing the locked v1.
V2 = ROOT / "outputs" / "v2_coach" / "predictions_2026_27_v2_EXPLORATORY.csv"
CACHE = ROOT / "data" / "raw" / "standings"
OUT = ROOT / "docs" / "data.json"
SEASON = "2026-27"
GAMES = 82


def standings(day: date) -> pd.DataFrame:
    path = CACHE / f"standings_{SEASON}_{day.isoformat()}.csv"
    if path.exists():
        return pd.read_csv(path)
    df = _retry(lambda: leaguestandingsv3.LeagueStandingsV3(
        season=SEASON, season_type="Regular Season", timeout=60).get_data_frames()[0], f"standings {day}")
    df = df[["TeamID", "TeamCity", "TeamName", "Conference", "PlayoffRank", "WINS", "LOSSES"]]
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def r(x, nd=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def build(pred: pd.DataFrame, st: pd.DataFrame) -> dict:
    abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    st = st.assign(team=st["TeamID"].map(abbr), name=st["TeamCity"] + " " + st["TeamName"])
    df = pred.merge(st[["team", "name", "PlayoffRank", "WINS", "LOSSES"]], on="team", how="left", validate="1:1")
    assert len(df) == 30 and df["WINS"].notna().all(), f"unmatched teams: {df.loc[df['WINS'].isna(), 'team'].tolist()}"

    df["gp"] = df["WINS"] + df["LOSSES"]
    played = df["gp"] > 0
    df["pace_wins"] = (df["WINS"] / df["gp"] * GAMES).where(played)
    df["diff"] = df["pace_wins"] - df["predicted_wins"]
    df["in_pi"] = ((df["pace_wins"] >= df["pi80_low"]) & (df["pace_wins"] <= df["pi80_high"])).where(played)

    conferences = {}
    for conf, g in df.groupby("conference"):
        gp = g[g["gp"] > 0]
        # Spearman needs >= 3 teams with some spread in actual win%; otherwise it's undefined.
        ok = len(gp) >= 3 and (gp["WINS"] / gp["gp"]).nunique() > 1
        rho = spearmanr(gp["predicted_wins"], gp["WINS"] / gp["gp"]).statistic if ok else None
        rho_v2 = spearmanr(gp["v2_wins"], gp["WINS"] / gp["gp"]).statistic if ok else None
        g = g.sort_values(["PlayoffRank" if played.any() else "predicted_conf_rank", "predicted_conf_rank"])
        conferences[conf] = {
            "spearman": r(rho, 3),
            "spearman_v2": r(rho_v2, 3),
            "teams": [{
                "team": t.team, "name": t.name, "actual_rank": int(t.PlayoffRank) if played.any() else None,
                "predicted_rank": int(t.predicted_conf_rank), "wins": int(t.WINS), "losses": int(t.LOSSES),
                "predicted_wins": r(t.predicted_wins), "v2_wins": r(t.v2_wins), "pi80_low": r(t.pi80_low), "pi80_high": r(t.pi80_high),
                "pace_wins": r(t.pace_wins), "diff": r(t.diff),
                "in_pi": None if pd.isna(t.in_pi) else bool(t.in_pi),
            } for t in g.itertuples()],
        }

    p = df[played]
    return {
        "season": SEASON,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "games_played": int(df["gp"].sum() // 2),
        "teams_with_games": int(played.sum()),
        "mae": r((p["pace_wins"] - p["predicted_wins"]).abs().mean() if len(p) else None, 2),
        "mae_v2": r((p["pace_wins"] - p["v2_wins"]).abs().mean() if len(p) else None, 2),
        "pi80_coverage": r(p["in_pi"].astype(bool).mean() if len(p) else None, 3),
        "conferences": conferences,
    }


def main():
    day = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()  # optional: replay a cached day
    pred = pd.read_csv(PRED).merge(pd.read_csv(V2)[["team", "v2_wins"]], on="team", how="left", validate="1:1")
    assert pred["v2_wins"].notna().all() and abs(pred["v2_wins"].sum() - 1230) < 1e-6
    data = build(pred, standings(day))
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {data['games_played']} games, MAE {data['mae']}, "
          f"Spearman E {data['conferences']['East']['spearman']} / W {data['conferences']['West']['spearman']}")


if __name__ == "__main__":
    main()
