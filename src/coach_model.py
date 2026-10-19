"""v2 exploratory: does a "coach ridge effect" improve on the locked v1 hybrid model?

Coach ridge effect: ridge regression of v1 win% residuals on one-hot starting-coach indicators (no intercept).
With one indicator per row, X'X is diagonal, so the ridge solution is closed form:
    effect_c = sum(residuals of coach c) / (n_seasons_c + lambda)
i.e. lambda = pseudo-seasons of shrinkage toward 0 (league average). A coach absent from the fit gets 0.

Leakage control: in every outer LOSO fold, the v1 OLS, the coach effects AND the choice of lambda
(an inner LOSO over the 14 training seasons) use only seasons other than the held-out one.

This is NOT RAPM: true RAPM needs stint-level play-by-play. Results go to outputs/v2_coach/;
the locked v1 predictions remain the official ones.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import OUTPUTS, PREDICT_SEASON, PROCESSED
from model import SPECS, fit, score, to_wins

V1 = SPECS["hybrid"]
LAMBDAS = np.round(np.logspace(-0.5, 4, 37), 3)  # ~0.3 .. 10,000 pseudo-seasons (10,000 ~= "no coach effect")
FIXED_LAMBDAS = [1, 3, 10, 30, 100, 1000]  # diagnostic: outer-LOSO MAE if lambda were fixed instead of chosen
OUT = OUTPUTS / "v2_coach"


def load() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "features.csv")
    coaches = pd.read_csv(PROCESSED / "coaches.csv")
    df = df.merge(coaches, on=["season_end", "team"], how="left", validate="one_to_one")
    assert df["coach_id"].notna().all()
    return df


def coach_effects(train: pd.DataFrame, lam: float, drop_midseason: bool = False) -> tuple:
    """Fit v1 OLS on train, then ridge coach effects on its residuals. Returns (ols result, effects in win%)."""
    res = fit(train, V1)
    resid = train["win_pct"] - res.fittedvalues
    rows = ~train["midseason_change"] if drop_midseason else pd.Series(True, index=train.index)
    g = resid[rows].groupby(train.loc[rows, "coach_id"])
    return res, g.sum() / (g.count() + lam)


def predict(res, effects: pd.Series, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(v1 win%, coach effect win%) for test rows; unseen coaches get 0."""
    base = res.predict(sm.add_constant(test[V1], has_constant="add"))
    eff = test["coach_id"].map(effects).fillna(0.0)
    return base, eff


def choose_lambda(train: pd.DataFrame, drop_midseason: bool) -> float:
    """Inner leave-one-season-out over the training seasons only; minimize rescaled-wins MAE."""
    err = np.zeros(len(LAMBDAS))
    for s in train["season_end"].unique():
        held, rest = train[train["season_end"] == s], train[train["season_end"] != s]
        res = fit(rest, V1)
        resid = rest["win_pct"] - res.fittedvalues
        rows = ~rest["midseason_change"] if drop_midseason else slice(None)
        g = resid[rows].groupby(rest.loc[rows, "coach_id"] if drop_midseason else rest["coach_id"])
        sums, counts = g.sum(), g.count()
        base = res.predict(sm.add_constant(held[V1], has_constant="add"))
        actual = held["win_pct"] * 82
        for i, lam in enumerate(LAMBDAS):
            wins, _ = to_wins(base + held["coach_id"].map(sums / (counts + lam)).fillna(0.0))
            err[i] += (wins - actual).abs().mean()
    return float(LAMBDAS[err.argmin()])


def loso(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, lams = [], []
    for s in sorted(train["season_end"].unique()):
        held, rest = train[train["season_end"] == s], train[train["season_end"] != s]
        preds = {}
        for name, drop in [("v1_coach", False), ("v1_coach_no_midseason", True)]:
            lam = choose_lambda(rest, drop)
            res, eff = coach_effects(rest, lam, drop)
            base, e = predict(res, eff, held)
            preds["v1"], preds[name] = base, base + e
            lams.append({"season_end": s, "variant": name, "lambda": lam,
                         "held_out_coaches_unseen": int((~held["coach_id"].isin(eff.index)).sum())})
        for name, p in preds.items():
            wins, _ = to_wins(p)
            rows.append({"season_end": s, "model": name, **score(held, wins)})
    return pd.DataFrame(rows), pd.DataFrame(lams)


def summarize(cv: pd.DataFrame) -> pd.DataFrame:
    s = cv.groupby("model").agg(mae=("mae", "mean"), spearman_east=("spearman_East", "mean"),
                                spearman_west=("spearman_West", "mean"))
    s["spearman_min"] = cv.groupby("model")[["spearman_East", "spearman_West"]].min().min(axis=1)
    piv = cv.pivot(index="season_end", columns="model", values="mae")
    s["seasons_beating_v1"] = [int((piv[m] < piv["v1"]).sum()) if m != "v1" else np.nan for m in s.index]
    return s.loc[["v1", "v1_coach", "v1_coach_no_midseason"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    train, future = df[df["season_end"] < PREDICT_SEASON], df[df["season_end"] == PREDICT_SEASON]

    cv, lams = loso(train)
    cv.to_csv(OUT / "cv_results.csv", index=False)
    lams.to_csv(OUT / "lambda_by_fold.csv", index=False)
    summary = summarize(cv)
    v1_locked = json.loads((OUTPUTS / "cv_summary.json").read_text())["cv"]["hybrid"]["mae"]
    assert abs(summary.loc["v1", "mae"] - v1_locked) < 1e-3, "v1 LOSO MAE does not reproduce the locked run"
    print("LOSO CV (15 seasons, wins on 82-game scale):")
    print(summary.round(3).to_string())
    print("\nper-season MAE:")
    print(cv.pivot(index="season_end", columns="model", values="mae")[summary.index].round(2).to_string())
    print("\nlambda chosen per fold (inner CV):")
    print(lams.pivot(index="season_end", columns="variant", values="lambda").to_string())

    curve = []
    for lam in FIXED_LAMBDAS:
        maes = []
        for s in sorted(train["season_end"].unique()):
            held, rest = train[train["season_end"] == s], train[train["season_end"] != s]
            res, eff = coach_effects(rest, lam)
            base, e = predict(res, eff, held)
            wins, _ = to_wins(base + e)
            maes.append((wins - held["win_pct"] * 82).abs().mean())
        curve.append({"lambda": lam, "loso_mae": float(np.mean(maes))})
    curve = pd.DataFrame(curve)
    curve.to_csv(OUT / "fixed_lambda_curve.csv", index=False)
    print(f"\nfixed-lambda diagnostic (v1 alone = {summary.loc['v1', 'mae']:.3f}):")
    print(curve.round(3).to_string(index=False))

    # Full-sample effects (all 15 seasons) for the 2026-27 comparison
    lam = choose_lambda(train, drop_midseason=False)
    res, eff = coach_effects(train, lam)
    n = train.groupby("coach_id").agg(coach=("coach", "first"), seasons_in_data=("season_end", "size"))
    effects = n.assign(effect_wins=eff * 82).sort_values("effect_wins", ascending=False)
    effects.to_csv(OUT / "coach_effects_full_sample.csv")
    print(f"\nfull-sample lambda = {lam}; effect range {effects['effect_wins'].min():.1f} .. {effects['effect_wins'].max():.1f} wins")

    base, e = predict(res, eff, future)
    v1_wins, _ = to_wins(base)
    v2_wins, _ = to_wins(base + e)
    locked = pd.read_csv(OUTPUTS / "predictions_2026_27.csv").set_index("team")["predicted_wins"]
    assert np.allclose(v1_wins.values, future["team"].map(locked).values), "v1 does not reproduce locked predictions"

    cmp_ = future[["team", "conf", "coach", "coach_id", "seasons_overall"]].assign(
        coach_seasons_in_data=future["coach_id"].map(n["seasons_in_data"]).fillna(0).astype(int),
        coach_effect_wins=(e * 82).values, v1_wins=v1_wins.values, v2_wins=v2_wins.values)
    cmp_["delta"] = cmp_["v2_wins"] - cmp_["v1_wins"]
    for col in ["v1", "v2"]:
        cmp_[f"{col}_rank"] = cmp_.groupby("conf")[f"{col}_wins"].rank(ascending=False, method="first").astype(int)
    cmp_ = cmp_.sort_values(["conf", "v2_rank"])
    cmp_.to_csv(OUT / "predictions_2026_27_v2_EXPLORATORY.csv", index=False)
    assert abs(cmp_["v2_wins"].sum() - 1230) < 1e-6

    first_year = cmp_[cmp_["seasons_overall"] == 1]
    first_year.to_csv(OUT / "first_year_coaches_2026_27.csv", index=False)
    show = ["team", "coach", "seasons_overall", "coach_seasons_in_data", "coach_effect_wins", "v1_wins", "v2_wins",
            "delta", "v1_rank", "v2_rank"]
    print("\n2026-27 first-year head coaches:")
    print(first_year[show].round(2).to_string(index=False))
    print("\nall teams, v1 (locked) vs v1+coach (exploratory):")
    print(cmp_[show].round(2).to_string(index=False))

    (OUT / "summary.json").write_text(json.dumps({
        "full_sample_lambda": lam, "cv": summary.round(4).to_dict(orient="index"),
        "lambda_by_fold": lams.groupby("variant")["lambda"].describe().round(3).to_dict(orient="index")},
        indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
