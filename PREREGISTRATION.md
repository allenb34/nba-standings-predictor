# Pre-registration: NBA 2026-27 regular-season wins

> **PROVISIONAL.** Committed Mon Oct 19, 2026. These are the model's predictions, but the roster mapping is still under
> manual review. The final lock commit (tag `v1-lock`, before Tue Oct 20, 2026 12:00 PM PT) removes this notice and
> either confirms these numbers or replaces them.

**No 2026-27 game results were used.** All model inputs come from seasons through 2025-26, plus the 2026-27 rosters
as listed by stats.nba.com before opening night.

## Decision rule (committed in `d0760ac` before cross-validation was run)
1. Lock **hybrid** only if its LOSO MAE is lower than **both** the naive baseline and team-only.
2. Otherwise, if team-only's MAE is lower than naive, lock **team-only** and document the roster feature as planned v2.
3. If neither model beats naive, lock whichever model has the lower MAE and state the loss to the baseline plainly.
4. If the hybrid is not validated by Tue Oct 20, 2026 9:00 AM PT, it is dropped and rules 2–3 are applied to team-only vs naive.

**Result: rule 1. The hybrid model is locked.**

## Model
OLS on team win% for season t (`statsmodels`), 450 team-seasons (2011-12 … 2025-26).

| Feature | Definition |
|---|---|
| `net_rtg_prev` | Team net rating, season t-1 (nba_api `LeagueDashTeamStats`, Advanced) |
| `roster_bpm` | Σ w·BPM / Σ w over the season-t roster. w = player's t-1 minutes, BPM = player's t-1 BPM (Basketball-Reference) |
| `continuity` | Share of the team's t-1 minutes played by players on its season-t roster |
| `roster_age` | Σ w·age / Σ w, using the same weights, with age in season t |

Prediction: win% × 82, then an additive shift so the 30 teams sum to exactly 1,230.
Conference rank = order within East/West. The 80% interval is the OLS prediction interval (α = 0.2), with the same shift applied.

### Fixed assumptions (chosen before fitting, not tuned)
- **No t-1 row**: use the player's t-2 BPM if he has one, with weight = min(t-2 minutes, 500). Otherwise BPM = **−2.0** at weight **500**.
  The same rule applies in training and for 2026-27.
- BPM left blank by BBRef (tiny-minute players) is set to −2.0. Their weight is negligible.
- **Historical roster** = each player's *first* team in season t. BBRef team rows are chronological, which is asserted on
  Dončić 2024-25 (DAL→LAL) and Harden 2021-22 (BKN→PHI).
- **2026-27 roster** = nba_api `CommonTeamRoster(season="2026-27")`, 526 players including two-way contracts. Names are matched to BBRef
  by normalized name. 75 players are unmatched: 73 rookies, plus Mario Hezonja and Tacko Fall, who have no NBA minutes since 2021-22.
  All 75 get replacement level. Ron Holland is hand-mapped.

## Cross-validation: leave-one-season-out, 15 folds
MAE is in wins on an 82-game scale. Spearman ρ is predicted vs actual rank within each conference.

| Model | MAE | Spearman East (mean) | Spearman West (mean) | Worst conference-season ρ |
|---|---|---|---|---|
| **hybrid** | **6.89** | **0.73** | **0.71** | 0.36 |
| team-only | 7.69 | 0.62 | 0.60 | 0.28 |
| naive (0.6·last + 0.4·41) | 7.89 | 0.59 | 0.58 | 0.21 |

The hybrid's MAE is better than naive's in all 15 seasons, and better than team-only's in 14 of 15 (2015-16 is the exception: 8.08 vs 7.52). Per-season numbers are in `outputs/cv_results.csv`.

## Full-sample fit (hybrid)
| | coef | SE | p |
|---|---|---|---|
| const | 0.5325 | 0.104 | <0.001 |
| net_rtg_prev | −0.0011 | 0.002 | 0.58 |
| roster_bpm | 0.1089 | 0.010 | <0.001 |
| continuity | 0.1272 | 0.034 | <0.001 |
| roster_age | −0.0041 | 0.004 | 0.27 |

R² = 0.525. The VIFs for net_rtg_prev and roster_bpm are 4.2 and 4.5: the two overlap, because BPM is built from team ratings.
With roster BPM in the model, prior net rating adds nothing. It stays in the model because the spec was registered before fitting.
Full output is in `outputs/coefs_*.txt`, `outputs/vif.csv` and `outputs/residuals.png`.

## Scoring (applied after April 11, 2027)
- MAE of `predicted_wins` vs actual wins, for all 30 teams
- Spearman ρ of predicted vs actual conference rank, for East and West separately
- Share of teams whose actual wins fall inside the 80% interval (target ≈ 80%)

## Known limitations
- **Hindsight rosters in training.** Historical rosters are built from players who actually logged minutes in season t,
  each on his first team. Opening-night players who never played (for example, injured all season) are missing, and that leaks
  a little information compared with the true 2026-27 setup.
- **Rookies** are a flat −2.0 BPM at 500 minutes, so high draft picks are probably undervalued.
- **Returning-from-injury players** with no t-1 row use t-2 BPM, which may be stale.
- **Two-way players** are on the 2026-27 rosters. Their low prior minutes keep their weight small.
- **Shrinkage.** OLS pulls extreme teams toward the mean, as the in-sample residual plot shows. The 80% intervals are about ±11 wins.
- **Shortened seasons** (2011-12, 2019-20, 2020-21) are handled by using win% as the target.
