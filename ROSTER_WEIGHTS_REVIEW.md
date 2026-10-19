# Post-lock review: how player roles are weighted (v1.1 candidate)

**Branch `v1.1-roster-weights`. Prompted on Oct 19, 2026, after the lock, by questioning WAS (21.0), IND (32.8) and PHI (43.3).**
The locked v1 predictions (tag `v1-lock`) were not modified.

## What the review found
There's no code bug: the 2026-27 rosters are current, with LeBron James and Jaylen Brown on PHI, and Trae Young and
Anthony Davis on WAS. The weakness is a design flaw. v1 uses **each player's 2025-26 minutes as his role**, which means:
- **Stars returning from injury count like bench players.** Haliburton missed 2025-26, so the t-2 rule capped him at
  500 minutes (2.6% of IND's weight). Trae Young (384 min) and Anthony Davis (626) together are 5% of WAS. Embiid is 5.2% of PHI.
- **Bench players keep minutes they earned on other teams, with no competition for minutes.** PHI's roster is weighted for
  22,940 minutes against the roughly 19,800 a team actually plays. Rayan Rupert (−5.1 BPM) carries 1,072 of them.

## Test (decision rule committed in `83a8659` before running)
A full 2×2×2 grid of feature-construction variants: role = max(t-1, t-2 minutes), "healthy"; BPM = minutes-weighted
t-1/t-2 blend; only the top 10 players by weight. Model spec and validation are unchanged (OLS, 4 features, LOSO CV).

| Variant | LOSO MAE | Spearman E | Spearman W |
|---|---|---|---|
| **v1** (t-1 minutes, t-1 BPM, all players) | 6.885 | 0.726 | 0.709 |
| top 10 only | 6.932 | 0.711 | 0.670 |
| 2-yr BPM | 6.915 | 0.703 | 0.691 |
| 2-yr BPM + top 10 | 6.966 | 0.697 | 0.666 |
| healthy minutes | 6.900 | 0.705 | 0.693 |
| healthy + top 10 | 6.946 | 0.712 | 0.680 |
| healthy + 2-yr BPM | 6.844 | 0.714 | 0.691 |
| **healthy + 2-yr BPM + top 10 (v1.1)** | **6.811** | 0.721 | 0.677 |

The rule said to adopt the best variant if its MAE beats v1 by at least 0.05 wins. v1.1 beats it by 0.074, so it passes,
and its predictions are in `outputs/v1_1/`.

## But the improvement is not reliable
- v1.1 is better in **9 of 15** seasons and worse in 6. The Wilcoxon signed-rank test gives **p = 0.56**.
- Most of the gain comes from one season (2015-16: −1.94). It's clearly worse in 2023-24 (+1.34) and 2025-26 (+0.78).
- Each fix on its own made predictions *worse*. Only the full combination helps, which is the pattern you'd expect if
  the best of 7 tries won by chance.
- Spearman (rank accuracy) falls slightly, from 0.709 to 0.677 in the West.

**My call: the evidence doesn't show v1.1 is better than v1.** The 0.05-win threshold in the rule was too weak a bar,
because it didn't require consistency across seasons. A stricter rule would also have required something like
"better in at least 11 of 15 seasons".

## Effect on the questioned teams
| Team | v1 (locked) | v1.1 | Why |
|---|---|---|---|
| IND | 32.8 | **40.0** | Haliburton goes from 2.6% to 12.5% of the rotation. This is the clearest instance of the flaw. |
| WAS | 21.0 | 22.8 | Even with Young and Davis fully weighted, 6 of the top 10 have a BPM of −2.5 to −5.0. |
| PHI | 43.3 | 41.7 | Embiid played 1,775 minutes over two seasons. The supporting cast rates below average, and continuity is low (0.56). |

Other large moves: OKC 62.2 → 69.7, SAC 17.0 → 25.5, UTA 32.3 → 26.7, TOR 53.3 → 48.3.
The full table is in `outputs/v1_1/v1_vs_v1_1.csv`.

## Files
`src/features.py` (the `VARIANT` switches; the defaults reproduce v1 byte-for-byte), `src/weight_variants.py`, and in
`outputs/v1_1/`: `variants.csv`, `decision.json`, `features_v1_1.csv`, `predictions_2026_27_v1_1.csv`, `v1_vs_v1_1.csv`.
