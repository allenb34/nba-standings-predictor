"""Three data-driven reasons per team for the locked 2026-27 projections -> outputs/explanations_2026_27.csv.

Post-lock explanation only; predictions are not changed. Each projection is decomposed EXACTLY, relative to a team with
league-average features (~41 wins):
    wins = base + sum(contributions) + shift
  roster player i:  b_bpm * 82 * share_i * (bpm_i - avg roster BPM)   (summed over the top-10 rotation = roster term)
  net rating / continuity / age:  b * 82 * (x - training mean)
The reasons are the two largest contributions by absolute size, plus the largest contribution pulling against the
team's overall direction (at least 0.5 wins; otherwise the third-largest driver). The script asserts the decomposition reproduces
outputs/predictions_2026_27.csv to 1e-6 wins.
"""
import contextlib
import io

import numpy as np
import pandas as pd

import features as F
from model import SPECS, fit

OUT = F.OUTPUTS / "explanations_2026_27.csv"


def main():
    df = pd.read_csv(F.PROCESSED / "features.csv")
    train, future = df[df["season_end"] < F.PREDICT_SEASON], df[df["season_end"] == F.PREDICT_SEASON]
    cols = SPECS["hybrid"]
    b = fit(train, cols).params
    mean = train[cols].mean()
    base = (b["const"] + (b[cols] * mean).sum()) * 82
    locked = pd.read_csv(F.OUTPUTS / "predictions_2026_27.csv").set_index("team")
    shift = (1230 - (b["const"] + future[cols].values @ b[cols].values).sum() * 82) / 30

    totals, _ = F.load_players()
    with contextlib.redirect_stdout(io.StringIO()):
        ros = F.map_2026_27_roster(totals)
    by = totals.set_index(["season_end", "player_id"])

    rows = []
    for _, t in future.iterrows():
        team = t["team"]
        r = ros[ros["team"] == team]
        p = F.player_inputs(r["player_id"], F.PREDICT_SEASON, totals).assign(name=r["PLAYER"].values)
        p = p.nlargest(F.VARIANT["top_k"], "w")
        p["share"] = p["w"] / p["w"].sum()
        items = []
        for _, x in p.iterrows():
            c = b["roster_bpm"] * 82 * x["share"] * (x["bpm"] - mean["roster_bpm"])
            if x["src"] == "replacement":
                why = f"{x['name']}: no NBA track record, counted at replacement level (−2.0 BPM)"
            elif x["src"] == "t-2":
                why = f"{x['name']}: {x['bpm']:+.1f} BPM in 2024-25 (missed 2025-26), {x['share']:.0%} of the rotation"
            else:
                mp_last = by.loc[(F.PREDICT_SEASON - 1, x["player_id"]), "mp"]
                two = (F.PREDICT_SEASON - 2, x["player_id"]) in by.index
                boosted = x["w"] >= 1.15 * mp_last and x["w"] - mp_last >= 200
                label = "BPM over the last two seasons" if two else "BPM in 2025-26"
                note = f" (role based on his larger 2024-25 workload; {mp_last:.0f} min in 2025-26)" if boosted else ""
                why = f"{x['name']}: {x['bpm']:+.1f} {label}, {x['share']:.0%} of the rotation{note}"
            items.append((why, c))
        items += [
            (f"Last season's net rating of {t['net_rtg_prev']:+.1f} (league average 0.0)",
             b["net_rtg_prev"] * 82 * (t["net_rtg_prev"] - mean["net_rtg_prev"])),
            (f"{t['continuity']:.0%} of last season's minutes return (average {mean['continuity']:.0%})",
             b["continuity"] * 82 * (t["continuity"] - mean["continuity"])),
            (f"Rotation average age {t['roster_age']:.1f} (average {mean['roster_age']:.1f})",
             b["roster_age"] * 82 * (t["roster_age"] - mean["roster_age"])),
        ]
        total = base + sum(c for _, c in items) + shift
        assert abs(total - locked.loc[team, "predicted_wins"]) < 1e-6, (team, total, locked.loc[team, "predicted_wins"])
        # Two largest drivers, plus the biggest factor pulling the other way (why the team isn't even higher / lower).
        ranked = sorted(items, key=lambda i: -abs(i[1]))
        net = total - (base + shift)
        counter = [i for i in ranked[2:] if np.sign(i[1]) == -np.sign(net) and abs(i[1]) >= 0.5]
        top = ranked[:2] + (counter[:1] or ranked[2:3])
        for k, (why, c) in enumerate(top, 1):
            rows.append({"team": team, "rank": k, "reason": why, "wins_vs_average": round(c, 1), "base_wins": round(base + shift, 1),
                         "role": "counterweight" if k == 3 and counter else "driver"})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: 3 reasons x {out['team'].nunique()} teams; base (average team) = {base + shift:.1f} wins after rescale")
    for team in ["OKC", "IND", "WAS", "PHI", "SAC"]:
        print(f"\n{team} ({locked.loc[team, 'predicted_wins']:.1f}):")
        for _, x in out[out["team"] == team].iterrows():
            print(f"  {x['wins_vs_average']:+5.1f}  {x['reason']}")


if __name__ == "__main__":
    main()
