# Pre-registration: NBA 2026-27 regular-season wins

> **LOCKED: tag `v1.1-lock`, Mon Oct 19, 2026, about 20 hours before opening tip (Tue Oct 20, 2026 12:00 PM PT).**
> This **replaces** the original lock, `v1-lock` (commit `ab1c9f1`, pushed 1:57 PM PT the same day).
> The change was made *after* the v1 predictions had been seen. Read the disclosure below before relying on these numbers.

**No 2026-27 game results were used.** All model inputs come from seasons through 2025-26, plus the 2026-27 rosters
as listed by stats.nba.com before opening night.

## Disclosure: why the official predictions changed after `v1-lock`
1. **The trigger.** After the v1 lock, the author questioned three projections: Washington (21.0 wins), Indiana (32.8)
   and Philadelphia (43.3). The change was prompted by *looking at specific teams' predictions*. That is exactly the
   risk a pre-registration exists to guard against, so it's stated here plainly.
2. **What the review found.** There was no code bug, and the rosters were current. The flaw was in the design: v1 used each
   player's 2025-26 minutes as his role. That treated stars returning from injury as bench players. Tyrese Haliburton was capped
   at 500 minutes, 2.6% of Indiana's weight. Trae Young (384 min) and Anthony Davis (626) made up 5% of Washington's.
   Deep-bench players also kept minutes earned on other teams, with no competition for playing time.
3. **How the fix was chosen.** A fixed 2×2×2 grid of feature-construction variants was tested. The decision rule was committed
   and pushed **before** the test ran (commit `83a8659`, 3:16 PM PT): take the lowest-MAE variant if it beats v1 by
   ≥ 0.05 wins. The winner was role = max(t-1, t-2 minutes), BPM = minutes-weighted average of t-1 and t-2, top 10
   players only. Its LOSO MAE is **6.811 vs 6.885**. Full results are in
   [`ROSTER_WEIGHTS_REVIEW.md`](ROSTER_WEIGHTS_REVIEW.md) and `outputs/v1_1/variants.csv`.
4. **The caveat.** The improvement over v1 is **not statistically reliable**. v1.1 is better in 9 of 15 seasons, with
   Wilcoxon p = 0.56, and most of the gain comes from one season (2015-16). Each fix alone made predictions worse, and the
   0.05-win rule didn't require consistency across seasons. v1.1 was adopted because it removes a known structural
   flaw (injured stars undervalued) without being worse on average. It has not been shown to be better.
5. **What stays the same.** The model form (OLS, same 4 features), the target, the 1,230-win rescale, the 80% interval,
   the scoring rules, the data and the roster mapping are all unchanged. The original decision rule (below) was re-applied
   to the v1.1 features and again selects the hybrid model.
6. **Accountability.** The v1 predictions stay public at tag `v1-lock` and in `outputs/v1/predictions_2026_27_v1.csv`.
   **Both v1 and v1.1 will be scored and reported** at the end of the season, so readers can judge whether the change helped.

## Decision rule (committed in `d0760ac` before cross-validation was first run)
1. Lock **hybrid** only if its LOSO MAE is lower than **both** the naive baseline and team-only.
2. Otherwise, if team-only's MAE is lower than naive, lock **team-only** and document the roster feature as planned v2.
3. If neither model beats naive, lock whichever model has the lower MAE and state the loss to the baseline plainly.
4. If the hybrid is not validated by Tue Oct 20, 2026 9:00 AM PT, it is dropped and rules 2–3 are applied to team-only vs naive.

**Result (v1.1 features): rule 1. The hybrid model is locked.**

## Model
OLS on team win% for season t (`statsmodels`), 450 team-seasons (2011-12 … 2025-26).

| Feature | Definition |
|---|---|
| `net_rtg_prev` | Team net rating, season t-1 (nba_api `LeagueDashTeamStats`, Advanced) |
| `roster_bpm` | Σ w·BPM / Σ w over the **10 most heavily weighted** players on the season-t roster. w = role (see below), BPM = the player's blended BPM (Basketball-Reference) |
| `continuity` | Share of the team's t-1 minutes played by players on its season-t roster |
| `roster_age` | Σ w·age / Σ w over the same 10 players, with age in season t |

**Role weight and BPM (v1.1):**
- **Has a t-1 row**: w = max(t-1 minutes, t-2 minutes). BPM = the minutes-weighted average of t-1 and t-2 BPM (t-1 only if he has no t-2 row).
- **No t-1 row, but has a t-2 row** (missed the whole season): w = t-2 minutes, BPM = t-2 BPM.
- **Neither** (true rookies): BPM = **−2.0** at w = **500 minutes**.
- The same rules apply in training and for 2026-27.

Prediction: win% × 82, then an additive shift so the 30 teams sum to exactly 1,230.
Conference rank = order within East/West. The 80% interval is the OLS prediction interval (α = 0.2), with the same shift applied.

### Fixed assumptions
- BPM left blank by BBRef (tiny-minute players) is set to −2.0.
- **Historical roster** = each player's *first* team in season t. BBRef team rows are chronological, which is asserted on
  Dončić 2024-25 (DAL→LAL) and Harden 2021-22 (BKN→PHI).
- **2026-27 roster** = nba_api `CommonTeamRoster(season="2026-27")`, 526 players including two-way contracts. Names are matched to BBRef
  by normalized name. 75 players are unmatched: 73 rookies, plus Mario Hezonja and Tacko Fall, who have no NBA minutes since 2021-22.
  All 75 get replacement level. Ron Holland is hand-mapped.

## Cross-validation: leave-one-season-out, 15 folds (v1.1 features)
MAE is in wins on an 82-game scale. Spearman ρ is predicted vs actual rank within each conference.

| Model | MAE | Spearman East (mean) | Spearman West (mean) | Worst conference-season ρ |
|---|---|---|---|---|
| **hybrid (v1.1, official)** | **6.81** | **0.72** | **0.68** | 0.35 |
| team-only | 7.69 | 0.62 | 0.59 | 0.26 |
| naive (0.6·last + 0.4·41) | 7.89 | 0.59 | 0.58 | 0.21 |
| *for reference: hybrid v1 (`v1-lock`)* | *6.89* | *0.73* | *0.71* | *0.36* |

The v1.1 hybrid's MAE is better than both naive's and team-only's in all 15 seasons. Compared with v1, its MAE is lower, but
its Spearman is slightly lower in both conferences (East 0.721 vs 0.726, West 0.677 vs 0.709). Per-season numbers are in `outputs/cv_results.csv`.

## Full-sample fit (hybrid, v1.1)
| | coef | SE | p |
|---|---|---|---|
| const | 0.7137 | 0.104 | <0.001 |
| net_rtg_prev | 0.0045 | 0.002 | 0.012 |
| roster_bpm | 0.0880 | 0.009 | <0.001 |
| continuity | 0.1556 | 0.035 | <0.001 |
| roster_age | −0.0124 | 0.004 | 0.001 |

R² = 0.510. Every coefficient now has the expected sign and is significant. In v1, prior net rating was
wrong-signed and insignificant. VIFs are 2.9 (net rating), 3.5 (roster BPM), 1.1 (continuity) and 1.6 (age).
Full output is in `outputs/coefs_*.txt`, `outputs/vif.csv` and `outputs/residuals.png`.

## Scoring (applied after April 11, 2027)
- MAE of `predicted_wins` vs actual wins, for all 30 teams
- Spearman ρ of predicted vs actual conference rank, for East and West separately
- Share of teams whose actual wins fall inside the 80% interval (target ≈ 80%)
- The same three metrics reported for the superseded v1 predictions, side by side

## Known limitations
- **Post-hoc revision.** See the disclosure. The v1.1 change was motivated by specific predictions, and its validation
  gain isn't statistically reliable.
- **Hindsight rosters in training.** Historical rosters are built from players who actually logged minutes in season t,
  each on his first team. Opening-night players who never played (for example, injured all season) are missing.
- **Healthy-minutes role** assumes a player returns to his larger recent workload. That overrates chronically injured
  or declining players. Embiid is the opposite case: he has played little in both seasons, so he still carries a low weight.
- **Rookies** are a flat −2.0 BPM at 500 minutes, so high draft picks are probably undervalued.
- **Shrinkage.** OLS pulls extreme teams toward the mean. The 80% intervals are about ±11 wins.
- **Shortened seasons** (2011-12, 2019-20, 2020-21) are handled by using win% as the target.
