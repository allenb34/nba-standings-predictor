# Post-lock exploratory analysis: does a head-coach effect help?

**Branch `v2-coach-effects`. Exploratory only.** The locked v1 predictions (tag `v1-lock`,
`outputs/predictions_2026_27.csv`) remain the official scored predictions all season. Nothing in v1 was modified.

## Result
**No. The coach effect adds nothing.** With λ chosen honestly inside each fold, the data shrinks every coach effect
to essentially zero. LOSO MAE is 6.891 for v1 + coach vs 6.885 for v1 alone, and every way of forcing a real coach
effect into the model makes predictions *worse*.

## 1. Is there a public coach RAPM dataset? No usable one.
- **Jeremias Engelmann's "Coach RAPM"** exists. It enters the head coach as a "6th player" in stint-level RAPM,
  so it's true RAPM. It's published as charts and X posts and in a paywalled
  [5x5 / Royce Webb post](https://www.roycewebb.com/p/coach-rapm-which-nba-coaches-actually),
  and it's discussed on [APBRmetrics](https://apbr.org/metrics/viewtopic.php?t=10074) and cited in a
  [UNC paper](https://www.unc.edu/wp-content/uploads/2025/06/coach_research.pdf).
  I found **no downloadable dataset and no per-season values**, only career-level ratings for current coaches.
- Other work estimates season-level or fixed effects in research papers but doesn't publish a maintained dataset:
  [Cannon et al. 2025, JQAS](https://www.degruyterbrill.com/document/doi/10.1515/jqas-2025-0025/html),
  [fixed-effects tutorial](https://towardsdatascience.com/quantifying-the-contribution-of-nba-coaches-using-fixed-effects-56f77f22153a),
  [Berry & Fowler via MIT Sloan](https://sloanreview.mit.edu/?p=66496),
  and [Chance on coaching tenure](https://chance.amstat.org/?p=11051).
- Even if Engelmann's numbers were downloadable, career ratings fit on 2025-26 data would leak into the
  held-out seasons of a leave-one-season-out CV. So I built the measure below instead.

## 2. Method: "coach ridge effect" (not RAPM)
- **Coach** = the head coach who **started** each season, taken from the Basketball-Reference season coaches pages
  (`src/coaches.py`, cached, 4 s between requests). Rows within a team are chronological, which is asserted on
  2017-18 MEM (Fizdale → Bickerstaff) and 2025-26 NOP (Green → Borrego).
  40 of 450 team-seasons had a midseason change. They're flagged and still credited to the starting coach.
- **Effect**: a ridge regression of v1 win% residuals on one-hot coach indicators. With one indicator per row, it has a
  closed form: *effect = Σ residuals ÷ (seasons + λ)*, where λ is the number of pseudo-seasons of shrinkage toward 0 (league average).
- **Leakage control**: in every outer LOSO fold, the v1 OLS, the coach effects, *and* λ (chosen by an inner
  leave-one-season-out over the 14 training seasons, from a grid of 0.3–10,000) use only the other seasons.
- **Coaches with no prior seasons in the training data** get an effect of 0 (league average). Across the 15 folds,
  16 held-out team-seasons had a coach the fold had never seen.
- **Sensitivity check**: refit with all midseason-change seasons excluded from the effect estimation.

## 3. v1 vs v1 + coach: LOSO CV, 15 seasons (same metrics as v1)
| Model | MAE (wins) | Spearman East | Spearman West | Worst conf-season ρ |
|---|---|---|---|---|
| **v1 (locked)** | **6.885** | **0.726** | **0.710** | 0.356 |
| v1 + coach ridge effect | 6.891 | 0.723 | 0.708 | 0.356 |
| v1 + coach, midseason changes excluded | 6.893 | 0.724 | 0.708 | 0.356 |

- **λ chosen by inner CV:** 10,000 (the top of the grid, meaning "no coach effect") in 14 of 15 folds. In the 2013-14 fold
  it chose 75, and that fold's MAE got worse (6.74 → 6.84), which accounts for nearly all of the gap above.
  The other 14 folds differ from v1 by less than 0.002 wins.
- **Forcing a coach effect in** with λ held fixed rather than chosen makes things steadily worse:

  | λ (pseudo-seasons) | 1 | 3 | 10 | 30 | 100 | 1000 | v1 alone |
  |---|---|---|---|---|---|---|---|
  | LOSO MAE | 7.33 | 7.15 | 6.98 | 6.92 | 6.90 | 6.886 | 6.885 |

**Interpretation**: once roster quality (minutes-weighted BPM), continuity, age and prior net rating are in the model,
what's left over for a coach doesn't carry over from one season to the next. The data can't tell a coach effect apart
from noise. The median coach has only 4 seasons in the window, and 16 coaches have just 1, so this doesn't prove
coaches don't matter. It shows that a season-level residual effect can't be estimated well enough to improve these predictions.

## 4. 2026-27 first-year head coaches
Their effect is 0 by construction (no prior seasons). Their teams move only through the 1,230-win rescale:

| Team | Coach | Coach effect | v1 (locked) | v1 + coach | Δ | Conf rank v1 → v2 |
|---|---|---|---|---|---|---|
| ORL | Sean Sweeney | 0.0 | 46.77 | 46.77 | −0.001 | 6 → 6 |
| POR | Micah Nori | 0.0 | 44.84 | 44.84 | −0.001 | 6 → 6 |
| DAL | Dusty May | 0.0 | 33.92 | 33.92 | −0.001 | 11 → 11 |

Across all 30 teams the largest change is under 0.005 wins (0.0043), and no conference rank changes
(`outputs/v2_coach/predictions_2026_27_v2_EXPLORATORY.csv`).
Tiago Splitter (CHI) also gets 0. He coached POR in 2025-26, but Chauncey Billups started that season, so the
starting-coach rule never credits Splitter with a season.

## Limitations
- Season-level only: true RAPM would need stint-level play-by-play. Residuals also carry injuries, luck and trades.
- A coach fired early in the season is credited with the whole season. Excluding those seasons doesn't change the result.
- Coaches and rosters are confounded: good coaches get good rosters, and the roster features absorb some of that.
- The 2011-12 … 2025-26 window gives few seasons per coach.

## Files
`src/coaches.py`, `src/coach_model.py`, `data/processed/coaches.csv`, `data/raw/coaches_*.csv`, and in `outputs/v2_coach/`:
`cv_results.csv`, `lambda_by_fold.csv`, `fixed_lambda_curve.csv`, `coach_effects_full_sample.csv`,
`first_year_coaches_2026_27.csv`, `predictions_2026_27_v2_EXPLORATORY.csv`, `summary.json`.

Reproduce: `python src/coaches.py && python src/coach_model.py`.
The script asserts that it reproduces v1's locked LOSO MAE and locked 2026-27 predictions.
