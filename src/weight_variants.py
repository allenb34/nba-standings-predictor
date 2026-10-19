"""v1.1 roster-weighting review: test fixes for how player roles are weighted in roster BPM and roster age.

Motivation (found after the v1 lock, prompted by WAS / IND / PHI looking too low): v1 uses each player's t-1 minutes
as his role, so stars returning from injury (Haliburton: capped at 500; Trae Young 384; Anthony Davis 626) count like
bench players, and deep-bench players carry large t-1 minutes from other teams with no competition for minutes.

Variants, a full 2 x 2 x 2 grid fixed before running (see VARIANT in features.py):
  weight: t1 (v1) | healthy  = role from max(t-1, t-2 minutes); t-2-only players keep full t-2 minutes
  bpm:    t1 (v1) | blend2   = minutes-weighted average of t-1 and t-2 BPM
  top_k:  None (v1) | 10     = only the 10 heaviest-weighted players count

DECISION RULE (committed before this script was first run):
  Among the 7 non-v1 variants, take the one with the lowest hybrid LOSO MAE. Adopt it as v1.1 only if its MAE is at
  least 0.05 wins lower than v1's (6.885). Otherwise v1 stands and these results are reported as a null finding.
  The model spec (OLS, same 4 features, same hybrid/team-only/naive comparison) is unchanged; only feature construction varies.
"""
import contextlib
import io
import itertools
import json

import pandas as pd

import features as F
from model import loso_cv, predict, sanity, summarize

OUT = F.OUTPUTS / "v1_1"
V1_MAE = 6.885
MIN_GAIN = 0.05


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    totals, per_team = F.load_players()
    teams = F.load_teams()
    with contextlib.redirect_stdout(io.StringIO()):
        ros = F.map_2026_27_roster(totals)

    rows, built = [], {}
    for weight, bpm, top_k in itertools.product(["t1", "healthy"], ["t1", "blend2"], [None, 10]):
        F.VARIANT.update(weight=weight, bpm=bpm, top_k=top_k)
        name = f"weight={weight}|bpm={bpm}|top_k={top_k}"
        df = F.build_features(totals, per_team, teams, ros)
        s = summarize(loso_cv(df[df["season_end"] < F.PREDICT_SEASON])).loc["hybrid"]
        rows.append({"variant": name, "is_v1": (weight, bpm, top_k) == ("t1", "t1", None), **s.round(4).to_dict()})
        built[name] = df
        print(f"{name:42s} MAE {s['mae']:.3f}  rho E {s['spearman_east']:.3f}  W {s['spearman_west']:.3f}")

    res = pd.DataFrame(rows).sort_values("mae")
    res.to_csv(OUT / "variants.csv", index=False)
    v1 = res[res["is_v1"]].iloc[0]
    assert abs(v1["mae"] - V1_MAE) < 1e-3, "v1 variant does not reproduce the locked LOSO MAE"
    best = res[~res["is_v1"]].iloc[0]
    adopt = best["mae"] <= V1_MAE - MIN_GAIN
    print(f"\nbest non-v1: {best['variant']}  MAE {best['mae']:.3f} vs v1 {V1_MAE}  -> "
          f"{'ADOPT as v1.1' if adopt else 'NOT adopted (gain < 0.05 wins); v1 stands'}")
    (OUT / "decision.json").write_text(json.dumps({
        "rule": f"lowest-MAE non-v1 variant, adopted only if MAE <= {V1_MAE} - {MIN_GAIN}",
        "best_variant": best["variant"], "best_mae": float(best["mae"]), "v1_mae": V1_MAE, "adopted": bool(adopt)},
        indent=2), encoding="utf-8")
    if not adopt:
        return

    df = built[best["variant"]]
    df.to_csv(OUT / "features_v1_1.csv", index=False)
    train, future = df[df["season_end"] < F.PREDICT_SEASON], df[df["season_end"] == F.PREDICT_SEASON]
    preds = predict(train, future, "hybrid")
    path = OUT / "predictions_2026_27_v1_1.csv"
    preds.to_csv(path, index=False)
    sanity(path)

    locked = pd.read_csv(F.OUTPUTS / "predictions_2026_27.csv").set_index("team")
    cmp_ = preds.set_index("team")[["conference", "predicted_wins", "predicted_conf_rank", "pi80_low", "pi80_high"]].join(
        locked[["predicted_wins", "predicted_conf_rank"]], rsuffix="_v1")
    cmp_["delta"] = cmp_["predicted_wins"] - cmp_["predicted_wins_v1"]
    cmp_ = cmp_.sort_values(["conference", "predicted_conf_rank"])
    cmp_.to_csv(OUT / "v1_vs_v1_1.csv")
    print("\n2026-27: v1 (locked) vs v1.1")
    print(cmp_[["predicted_wins_v1", "predicted_wins", "delta", "predicted_conf_rank_v1", "predicted_conf_rank"]].round(1).to_string())


if __name__ == "__main__":
    main()
