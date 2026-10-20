"""Post-lock check: are the 80% prediction intervals right, and should their widths vary by team?

The locked intervals are OLS prediction intervals: half-width = t * sqrt(sigma^2 + x'Cov(b)x). sigma (~8.5 wins)
dominates, so every team gets about +/-11. This tests whether that is calibrated out of sample, and whether
team-specific widths would be better.

Everything below is evaluated by leave-one-season-out (15 folds) on the official v1.1 features; held-out seasons never
inform their own intervals. Methods compared:
  A. official   OLS prediction interval (as locked).
  B. conformal  constant half-width = 80th percentile of |out-of-sample residual| from an inner LOSO over the training seasons.
  C. scaled     normalized conformal: |inner-LOSO residual| is regressed (OLS) on [continuity, |pred - 41|, n_replacement];
                half-width = q * s_hat(x), with q the 80th percentile of |resid| / s_hat on the training seasons.
                s_hat is floored at 25% of its training mean.
Scored with the interval score at alpha = 0.2 (Gneiting & Raftery 2007; lower is better):
  IS = width + (2/alpha)*(lo - y)*[y < lo] + (2/alpha)*(y - hi)*[y > hi]
which rewards narrow intervals and penalizes misses.

DECISION RULE (committed before this script was first run):
  B or C replaces A only if, versus A, it (1) has a lower interval score in at least 11 of 15 held-out seasons AND
  (2) has pooled held-out coverage between 75% and 85%. If both B and C qualify, the one with the lower mean interval score wins.
  If neither qualifies, the locked intervals stand. Point predictions are not touched by any method.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import OUTPUTS, PREDICT_SEASON, PROCESSED
from model import SPECS, fit

FEATS = SPECS["hybrid"]
ALPHA = 0.2
Z = ["continuity", "dist41", "n_replacement"]
OUT = OUTPUTS / "intervals"


def to_wins(win_pct: pd.Series) -> tuple[pd.Series, float]:
    wins = win_pct * 82
    return wins, (1230 - wins.sum()) / len(wins) if len(wins) == 30 else 0.0


def ols_pi(train, test):
    """Official method: point + OLS 80% interval, in wins, shifted so the 30 teams sum to 1,230."""
    f = fit(train, FEATS).get_prediction(sm.add_constant(test[FEATS], has_constant="add")).summary_frame(alpha=ALPHA)
    pred, shift = to_wins(f["mean"])
    return pred + shift, f["obs_ci_lower"] * 82 + shift, f["obs_ci_upper"] * 82 + shift


def inner_residuals(train):
    """Out-of-sample residuals (wins) for every training row, via LOSO over the training seasons."""
    rows = []
    for s in train["season_end"].unique():
        held, rest = train[train["season_end"] == s], train[train["season_end"] != s]
        pred, _, _ = ols_pi(rest, held)
        rows.append(held.assign(pred=pred.values, resid=(held["win_pct"] * 82 - pred).values))
    r = pd.concat(rows)
    return r.assign(dist41=(r["pred"] - 41).abs())


def intervals(train, test):
    """Return a frame with the point prediction and lo/hi for methods A, B, C."""
    pred, lo_a, hi_a = ols_pi(train, test)
    out = pd.DataFrame({"pred": pred.values, "lo_A": lo_a.values, "hi_A": hi_a.values}, index=test.index)
    r = inner_residuals(train)
    half_b = np.quantile(r["resid"].abs(), 1 - ALPHA)
    out["lo_B"], out["hi_B"] = out["pred"] - half_b, out["pred"] + half_b
    sm_fit = sm.OLS(r["resid"].abs(), sm.add_constant(r[Z])).fit()
    floor = 0.25 * r["resid"].abs().mean()
    s_tr = sm_fit.fittedvalues.clip(lower=floor)
    q = np.quantile(r["resid"].abs() / s_tr, 1 - ALPHA)
    zt = test.assign(dist41=(out["pred"] - 41).abs())
    s_te = sm_fit.predict(sm.add_constant(zt[Z], has_constant="add")).clip(lower=floor)
    out["lo_C"], out["hi_C"] = out["pred"] - q * s_te, out["pred"] + q * s_te
    return out, sm_fit


def interval_score(y, lo, hi):
    return (hi - lo) + (2 / ALPHA) * (lo - y) * (y < lo) + (2 / ALPHA) * (y - hi) * (y > hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED / "features.csv")
    train, future = df[df["season_end"] < PREDICT_SEASON], df[df["season_end"] == PREDICT_SEASON]

    rows = []
    for s in sorted(train["season_end"].unique()):
        held, rest = train[train["season_end"] == s], train[train["season_end"] != s]
        iv, _ = intervals(rest, held)
        y = held["win_pct"] * 82
        for m in "ABC":
            lo, hi = iv[f"lo_{m}"], iv[f"hi_{m}"]
            rows.append({"season_end": s, "method": m, "coverage": float(((y >= lo) & (y <= hi)).mean()),
                         "mean_width": float((hi - lo).mean()), "width_sd": float((hi - lo).std()),
                         "interval_score": float(interval_score(y, lo, hi).mean())})
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "interval_cv.csv", index=False)
    summ = cv.groupby("method").agg(coverage=("coverage", "mean"), mean_width=("mean_width", "mean"),
                                    width_sd=("width_sd", "mean"), interval_score=("interval_score", "mean"))
    piv = cv.pivot(index="season_end", columns="method", values="interval_score")
    summ["seasons_better_than_A"] = [np.nan if m == "A" else int((piv[m] < piv["A"]).sum()) for m in summ.index]
    print("Held-out (LOSO) 80% intervals, wins:")
    print(summ.round(3).to_string())

    ok = [m for m in "BC" if summ.loc[m, "seasons_better_than_A"] >= 11 and 0.75 <= summ.loc[m, "coverage"] <= 0.85]
    winner = min(ok, key=lambda m: summ.loc[m, "interval_score"]) if ok else "A"
    print(f"\nqualifying alternatives: {ok or 'none'} -> {'locked intervals stand' if winner == 'A' else 'method ' + winner}")

    iv, sm_fit = intervals(train, future)
    (OUT / "scale_model_full_sample.txt").write_text(sm_fit.summary().as_text(), encoding="utf-8")
    fut = future[["team"]].assign(**{c: iv[c].values for c in iv.columns})
    fut.to_csv(OUT / "intervals_2026_27_all_methods.csv", index=False)
    (OUT / "decision.json").write_text(json.dumps({"winner": winner, "qualifying": ok,
                                                    "summary": summ.round(4).to_dict(orient="index")}, indent=2, default=float),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()
