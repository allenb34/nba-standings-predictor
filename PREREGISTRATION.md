# Pre-registration: NBA 2026-27 regular-season wins

**Status: DRAFT. Decision rule committed before cross-validation was run. No predictions yet.**

## Decision rule (fixed before any CV result was seen)
Two OLS specs are evaluated by leave-one-season-out CV (MAE in wins, 82-game scale, 2011-12 … 2025-26):
- **hybrid**: prior net rating + minutes-weighted roster BPM + roster continuity + minutes-weighted roster age
- **team-only**: prior net rating + roster continuity + minutes-weighted roster age

Naive baseline: wins = 82 × (0.6 × last season's win% + 0.4 × 0.5).

1. Lock **hybrid** only if its LOSO MAE is lower than **both** the naive baseline and team-only.
2. Otherwise, if team-only's MAE is lower than naive, lock **team-only** and document the roster feature as planned v2.
3. If neither model beats naive, lock whichever model has the lower MAE and state the loss to the baseline plainly.
4. If the hybrid is not validated by Tue Oct 20, 2026 9:00 AM PT, it is dropped and rules 2–3 are applied to team-only vs naive.

Implemented in `decide()` in `src/model.py`.

## Fixed assumptions (chosen before fitting, not tuned)
- Replacement level BPM for players with no t-1 or t-2 data: **−2.0**
- Weight for players with no t-1 row: **500 minutes** (cap). A t-2 row supplies BPM, and its minutes are capped at 500.
- Historical roster = each player's **first** team in season t (BBRef rows are chronological; asserted on Dončić 2024-25 and Harden 2021-22).
- Wins are rescaled additively so the 30 teams sum to exactly 1,230.
