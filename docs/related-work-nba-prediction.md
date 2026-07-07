# Related Work: NBA Pre-Game Outcome Prediction

Deep-research summary (conducted 2026-07-06) on prior works — GitHub repos and papers —
predicting NBA game outcomes under the **same hard constraints this project follows**:

1. **Pre-game only** — features use only information available before tip-off.
2. **No betting/market data** as model inputs (odds may be used only as a benchmark, never as a feature).
3. **No data randomization** — train/test must respect chronological order (walk-forward / rolling-origin), never shuffled k-fold.

Golden rule: the *same* system must be able to run live alongside a season as it progresses.

## A. ~2/3 is near the real ceiling — and we are doing it right

- The NBA regular-season **upset rate is ~28–32%**: the "better" team wins only ~68–72% of
  the time, so pre-game accuracy is bounded around **~70%**. This figure is repeated
  independently across sources.
- "The best NBA game prediction models only accurately predict the winner about 70% of the time."
- **Tell that we're feature-limited, not model-limited:** our RF and LR converge on the *same*
  ~66–67%. When different model families land on the same number, the limit is in the
  information content of the features, not the algorithm. Swapping RF↔LR↔XGBoost won't move it.
- **Most "high" published accuracies are leakage.** 80–90% / 83–84% numbers come from
  in-game / third-quarter / post-game data (e.g., XGBoost+SHAP ~80% on 3rd-quarter data,
  >90% post-game). These violate the pre-game constraint and are not valid benchmarks.

## B. Constraint-aligned works, ranked

**Tier 1 — align well AND credibly beat 2/3 (sequence models):**

| Work | Acc. | Pre-game | Odds | Temporal split | Notes |
|---|---|---|---|---|---|
| Long-Sequence LSTM for NBA (arXiv 2512.08591, Dec 2025) | **72.35%** (AUC 0.76) | Likely (verify) | No | Multi-season 2004-05→2024-25 | LSTM over 8-season game sequences; beats its own LR/RF/MLP/CNN baselines |
| LSTM vs Transformer for NCAA (arXiv 2508.02725, 2025) | **73.3–73.6%** | Yes (explicit) | No | Yes (strict boundaries) | Transformer 73.63% / LSTM 73.31%; explicitly excludes outcome-dependent features. NCAA (tournament), not NBA |

Both suggest sequence models (LSTM/Transformer) consuming the game-by-game sequence buy a
genuine ~3–6 points over hand-pooled rolling averages — **if** built without target leakage
(the input window for game *t* must end at *t−1*).

**Tier 2 — align well, land right at our plateau (confirm the ceiling):**

| Work | Acc. | Notes |
|---|---|---|
| Pirkn/NBA-Game-Outcome-Prediction | NN 66.9%, Ridge 65.05%, XGBoost 64.8% | Elo + rolling avgs, 2016→present, no odds. Split unstated. Lists player/injury data as future work. |
| luke-lite/NBA-Prediction-Modeling | Elo 65.3%, GNB+PCA 63.5% | ~10 seasons, no odds, chronological. Cites the 68–72% ceiling; names injury data as next lever. |
| arttorres0/nba-games-predictor | ~66% | XGBoost, 2012–2018 box scores, no odds. |
| Bryant honors thesis | Simple Logistic ~69.67% | Full text not verified (403). |

**Flagged / not aligned:**

- Josh Weiner (Towards Data Science): RF **67.15%**, Elo + 10-game avgs + PER, no odds —
  **but random 80:20 split** leaks future→past; its temporally-valid number is likely ≤ our 66.77%.
- kyleskom/NBA-Machine-Learning-Sports-Betting (popular): betting-oriented, uses odds/EV — not
  aligned as a pure pre-game predictor.

## C. Diagnosis and highest-leverage next steps

**We are not doing something wrong.** Our methodology (walk-forward, `shift(1)`, no odds, no
shuffle, separate train/eval) is cleaner than much of the published/GitHub work — several
enviable numbers (e.g., 67.15%) get their edge partly from the leakage we correctly avoid.
Our 66.77% temporally-valid ≈ their headline numbers, honestly earned. We've saturated
team-level aggregate signal.

Ranked levers to break past 2/3 (all live-deployable):

1. **Player availability / injury features — the single biggest missing input.** Named as
   future work by both aligned repos. A star out swings a game ~3–7 points, and availability is
   known pre-game. Highest-credibility path from ~67% → ~69–70%. (Implemented in this project
   via the `InactivePlayers` dataset from BoxScoreSummaryV2/V3 — pre-tip declared inactives.)
2. **Player-based roster ratings** aggregated to the expected lineup (RAPM/RAPTOR-style or
   minutes-weighted player value) instead of team box-score aggregates.
3. **Sequence models (LSTM/Transformer)** over the per-team game sequence — Tier-1 papers
   suggest a real gain, but build leakage-safe and verify against the walk-forward baseline.
4. **Calibration** (Platt/isotonic on out-of-fold walk-forward probabilities) — improves
   Brier/log-loss (won't raise accuracy).

**Expectation setting:** the realistic frontier under these constraints is ~68–70%. Chasing
>72% pre-game/no-odds is likely chasing leakage. The win is squeezing 2–3 honest points via
player availability, not a leap.

## Sources

- Long-Sequence LSTM for NBA — https://arxiv.org/abs/2512.08591
- LSTM vs Transformer (NCAA) — https://arxiv.org/html/2508.02725v1
- Pirkn/NBA-Game-Outcome-Prediction — https://github.com/Pirkn/NBA-Game-Outcome-Prediction
- luke-lite/NBA-Prediction-Modeling — https://github.com/luke-lite/NBA-Prediction-Modeling
- arttorres0/nba-games-predictor — https://github.com/arttorres0/nba-games-predictor
- Josh Weiner (TDS) — https://towardsdatascience.com/predicting-the-outcome-of-nba-games-with-machine-learning-a810bb768f20/
- XGBoost+SHAP (in-game, leakage example) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11265715/
- (Mis)Use of ML with Panel Data — https://arxiv.org/pdf/2411.09218
- kyleskom/NBA-Machine-Learning-Sports-Betting — https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting
