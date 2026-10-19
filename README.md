# NBA 2026-27 Standings Predictor

Pre-registered regular-season win predictions for all 30 NBA teams, committed before opening tip on Oct 20, 2026.
The commit timestamp (and tag `v1-lock`) is the proof.

- Predictions: [`outputs/predictions_2026_27.csv`](outputs/predictions_2026_27.csv)
- Model spec, decision rule, CV results, limitations: [`PREREGISTRATION.md`](PREREGISTRATION.md)

## Method in one paragraph
OLS on team win% using four features: prior-season net rating, minutes-weighted roster BPM (each player's prior-season
Basketball-Reference BPM, weighted by his prior-season minutes), roster continuity, and minutes-weighted roster age.
It's trained on 450 team-seasons (2011-12 … 2025-26). Predicted win% × 82 is shifted so the league sums to 1,230 wins.
In leave-one-season-out CV it beats a naive baseline (0.6 × last season's wins + 0.4 × 41) on MAE, 6.89 vs 7.89 wins.

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
replacement level (−2.0 BPM). See [`PREREGISTRATION.md`](PREREGISTRATION.md) for the full list.

## Data
Basketball-Reference advanced tables (2009-10 … 2025-26), stats.nba.com via `nba_api` (team advanced stats, 2026-27 rosters).
