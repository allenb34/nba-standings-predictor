# NBA 2026-27 Standings Predictor

Pre-registered regular-season win predictions for all 30 NBA teams, committed before opening tip on Oct 20, 2026.
The commit timestamp (and tag `v1.1-lock`) is the proof.

**Live dashboard (predicted vs actual, updated weekly):** https://allenb34.github.io/nba-standings-predictor/

- Official predictions (v1.1): [`outputs/predictions_2026_27.csv`](outputs/predictions_2026_27.csv)
- Model spec, decision rule, CV results, limitations, **and the disclosure of the v1 → v1.1 revision**: [`PREREGISTRATION.md`](PREREGISTRATION.md)

## Revision before tip-off: v1 → v1.1
The first lock (`v1-lock`) used each player's prior-season minutes as his role, which undervalued stars returning from
injury, like Haliburton. A review prompted by questioning specific teams' projections led to a revised role weighting.
The test was pre-specified, and the fix was re-locked as `v1.1-lock` before tip-off:
[`ROSTER_WEIGHTS_REVIEW.md`](ROSTER_WEIGHTS_REVIEW.md). Its validation gain over v1 (LOSO MAE 6.81 vs 6.89) is
**not statistically reliable**. The superseded v1 predictions stay public in
[`outputs/v1/`](outputs/v1/predictions_2026_27_v1.csv), and both versions will be scored at season's end.

## Post-lock note: projections above 60 wins run high
Added after `v1.1-lock`. The locked predictions are unchanged. In leave-one-season-out validation, the model is well
calibrated up to about 60 wins: in every bin below 60, the average actual result is within 0.6 wins of the projection.
Its **60+ win projections have run about 3.5 wins high**, and 6 of those 7 teams finished below their projection.
So OKC's **69.7** is best read as a **mid-60s team with upside**, not a 70-win forecast.

## Post-lock: why each team is projected where it is
Each projection splits exactly into contributions relative to a team with league-average inputs (39.0 wins this
season): one per rotation player, plus net rating, continuity and age. [`src/explain.py`](src/explain.py) lists the two
biggest factors and the biggest one pulling the other way for every team, in
[`outputs/explanations_2026_27.csv`](outputs/explanations_2026_27.csv) and on the dashboard. It asserts that the parts
add back up to the locked projection. These are explanations only; no prediction changed.

## Post-lock: are the 80% intervals right?
Every interval is about ±11 wins because OLS assumes the same noise for every team, and that noise (about 8.5 wins)
dominates. A pre-registered check ([`src/interval_check.py`](src/interval_check.py), rule committed before running) found
the locked intervals **covered 81.1%** of held-out team-seasons, against a target of 80%. Team-specific widths, based on
continuity, distance from .500 and the number of rookies, did not beat them out of sample: better in only 6 of 15 seasons.
**The locked intervals stand.** The one known weakness is at the top: there the interval is about the right width, but it's
centered about 3.5 wins too high (see the note above). Results are in [`outputs/intervals/`](outputs/intervals/).

## Post-lock exploratory analysis (not official)
- **Coach effects:** [`COACH_ANALYSIS.md`](https://github.com/allenb34/nba-standings-predictor/blob/v2-coach-effects/COACH_ANALYSIS.md)
  on branch [`v2-coach-effects`](https://github.com/allenb34/nba-standings-predictor/tree/v2-coach-effects).
  Tests whether a head-coach "ridge effect" improves the model. Result: it doesn't. Cross-validation shrinks every coach effect to about zero.

## Method in one paragraph
OLS on team win% using four features: prior-season net rating, roster BPM (each player's two-season Basketball-Reference
BPM, weighted by his larger recent-season minutes, over the team's top 10 players), roster continuity, and
roster age. It's trained on 450 team-seasons (2011-12 … 2025-26). Predicted win% × 82 is shifted so the league sums
to 1,230 wins. In leave-one-season-out CV it beats a naive baseline (0.6 × last season's wins + 0.4 × 41) on MAE,
6.81 vs 7.89 wins, and does so in all 15 seasons.

## Reproduce
```bash
pip install -r requirements.txt
python src/fetch.py      # cache-first; with data/raw/*.csv present it makes 0 network calls
python src/features.py   # builds data/processed/features.csv, prints PHI/MIA spot checks + roster changes
python src/model.py      # LOSO CV, diagnostics, predictions, sanity asserts
```

## Limitations
Historical rosters are built from players who actually logged minutes in season t, each on his first team. That's mild
hindsight compared with an opening-day roster: players injured for the whole season are missing. Rookies are a flat
replacement level (−2.0 BPM). The v1.1 revision was made after seeing v1's predictions. See
[`PREREGISTRATION.md`](PREREGISTRATION.md) for the full list.

## Data
Basketball-Reference advanced tables (2009-10 … 2025-26), stats.nba.com via `nba_api` (team advanced stats, 2026-27 rosters).
