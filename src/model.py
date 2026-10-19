"""OLS fit, leave-one-season-out CV, diagnostics, 2026-27 prediction, sanity checks.

python src/model.py                   # apply the pre-registered decision rule
python src/model.py --model team_only # force a spec (used for the provisional lock)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

from features import OUTPUTS, PREDICT_SEASON, PROCESSED

SPECS = {
    "hybrid": ["net_rtg_prev", "roster_bpm", "continuity", "roster_age"],
    "team_only": ["net_rtg_prev", "continuity", "roster_age"],
}
TOTAL_WINS = 1230.0


def to_wins(win_pct: pd.Series) -> tuple[pd.Series, float]:
    """win% -> wins on an 82-game scale, shifted additively so the 30 teams sum to 1,230."""
    wins = win_pct * 82
    shift = (TOTAL_WINS - wins.sum()) / len(wins)
    return wins + shift, shift


def fit(df: pd.DataFrame, feats: list[str]):
    return sm.OLS(df["win_pct"], sm.add_constant(df[feats])).fit()


def score(season: pd.DataFrame, pred_wins: pd.Series) -> dict:
    actual = season["win_pct"] * 82  # 82-game scale so 66- and 72-game seasons compare
    out = {"mae": float((pred_wins - actual).abs().mean())}
    for conf in ["East", "West"]:
        m = season["conf"] == conf
        out[f"spearman_{conf}"] = float(spearmanr(pred_wins[m], actual[m]).statistic)
    return out


def loso_cv(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in sorted(train["season_end"].unique()):
        held, rest = train[train["season_end"] == s], train[train["season_end"] != s]
        preds = {name: fit(rest, f).predict(sm.add_constant(held[f], has_constant="add")) for name, f in SPECS.items()}
        preds["naive"] = 0.6 * held["win_pct_prev"] + 0.4 * 0.5
        for name, p in preds.items():
            wins, _ = to_wins(p)
            rows.append({"season_end": s, "model": name, **score(held, wins)})
    return pd.DataFrame(rows)


def summarize(cv: pd.DataFrame) -> pd.DataFrame:
    s = cv.groupby("model").agg(
        mae=("mae", "mean"),  # equal-size seasons, so mean of per-season MAE == pooled MAE
        spearman_east=("spearman_East", "mean"), spearman_west=("spearman_West", "mean"),
        spearman_min=("spearman_East", "min"))
    s["spearman_min"] = np.minimum(s["spearman_min"], cv.groupby("model")["spearman_West"].min())
    return s.loc[["hybrid", "team_only", "naive"]]


def decide(summary: pd.DataFrame) -> tuple[str, str]:
    """Pre-registered rule (PREREGISTRATION.md), written before CV was run."""
    h, t, n = summary.loc["hybrid", "mae"], summary.loc["team_only", "mae"], summary.loc["naive", "mae"]
    if h < n and h < t:
        return "hybrid", "hybrid beats both naive and team-only"
    if t < n:
        return "team_only", "team-only beats naive; hybrid does not beat both"
    best = "hybrid" if h < t else "team_only"
    return best, f"NEITHER model beats naive (naive MAE {n:.3f}); locking the better model, {best}"


def diagnostics(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(SPECS), figsize=(11, 4), sharey=True)
    vif_rows = []
    for ax, (name, feats) in zip(axes, SPECS.items()):
        res = fit(train, feats)
        (OUTPUTS / f"coefs_{name}.txt").write_text(res.summary().as_text(), encoding="utf-8")
        X = sm.add_constant(train[feats])
        vif_rows += [{"model": name, "feature": f, "vif": variance_inflation_factor(X.values, i)}
                     for i, f in enumerate(X.columns) if f != "const"]
        ax.scatter(res.fittedvalues * 82, res.resid * 82, s=8, alpha=0.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=f"{name} (in-sample)", xlabel="fitted wins (82-game scale)")
    axes[0].set_ylabel("residual (wins)")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "residuals.png", dpi=120)
    pd.DataFrame(vif_rows).to_csv(OUTPUTS / "vif.csv", index=False)


def predict(train: pd.DataFrame, future: pd.DataFrame, name: str) -> pd.DataFrame:
    feats = SPECS[name]
    res = fit(train, feats)
    frame = res.get_prediction(sm.add_constant(future[feats], has_constant="add")).summary_frame(alpha=0.2)
    wins, shift = to_wins(frame["mean"])
    out = pd.DataFrame({
        "team": future["team"].values,
        "conference": future["conf"].values,
        "predicted_wins": wins.values,
        "pi80_low": (frame["obs_ci_lower"] * 82 + shift).values,
        "pi80_high": (frame["obs_ci_upper"] * 82 + shift).values,
    })
    out["predicted_conf_rank"] = out.groupby("conference")["predicted_wins"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values(["conference", "predicted_conf_rank"])
    return out[["team", "conference", "predicted_wins", "predicted_conf_rank", "pi80_low", "pi80_high"]]


def sanity(path) -> None:
    p = pd.read_csv(path)
    assert abs(p["predicted_wins"].sum() - TOTAL_WINS) < 1e-6, p["predicted_wins"].sum()
    assert p["team"].nunique() == 30 and len(p) == 30
    assert (p["conference"].value_counts() == 15).all()
    assert (p["pi80_low"] < p["predicted_wins"]).all() and (p["predicted_wins"] < p["pi80_high"]).all()
    for _, r in p[(p["predicted_wins"] < 10) | (p["predicted_wins"] > 72)].iterrows():
        print(f"WARNING: {r['team']} predicted {r['predicted_wins']:.1f} wins (outside 10-72)")
    print("sanity checks passed: 1,230 wins, 30 teams, 15 per conference")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["auto", *SPECS], default="auto")
    args = ap.parse_args()

    df = pd.read_csv(PROCESSED / "features.csv")
    train, future = df[df["season_end"] < PREDICT_SEASON], df[df["season_end"] == PREDICT_SEASON]
    OUTPUTS.mkdir(exist_ok=True)

    cv = loso_cv(train)
    cv.to_csv(OUTPUTS / "cv_results.csv", index=False)
    summary = summarize(cv)
    print("\nLOSO CV (15 seasons, wins on 82-game scale):")
    print(summary.round(3).to_string())
    print("\nper-season MAE:")
    print(cv.pivot(index="season_end", columns="model", values="mae")[["hybrid", "team_only", "naive"]].round(2).to_string())

    diagnostics(train)
    rule_choice, reason = decide(summary)
    chosen = rule_choice if args.model == "auto" else args.model
    print(f"\ndecision rule -> {rule_choice}: {reason}")
    if chosen != rule_choice:
        print(f"OVERRIDE via --model: predicting with {chosen}")

    preds = predict(train, future, chosen)
    path = OUTPUTS / "predictions_2026_27.csv"
    preds.to_csv(path, index=False)
    sanity(path)
    (OUTPUTS / "cv_summary.json").write_text(json.dumps({
        "locked_model": chosen, "decision_rule_result": rule_choice, "reason": reason,
        "cv": summary.round(4).to_dict(orient="index")}, indent=2), encoding="utf-8")
    print(f"\n2026-27 predictions ({chosen}):")
    print(preds.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
