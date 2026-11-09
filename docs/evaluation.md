# Evaluation summary

Vartti estimates the price spread over the day-ahead spot price, then adds spot back to report price quantiles. It also estimates the probability of each regulation direction and of spreads above or below the exceptional-price thresholds. The team compared features on expanding time folds, enforced publication cutoffs and used validation data to select models and calibrate intervals.

| Metric | Challenge validation: persistence | Challenge validation: Blend B | Local test: Blend B |
| --- | ---: | ---: | ---: |
| Median price MAE (€/MWh) | 66.20 | 41.44 | 26.50 |
| Mean P10/P90 pinball loss | 24.18 | 11.23 | 6.96 |
| Mean exceptional-price Brier score | 0.1132 | 0.0871 | 0.0549 |
| Regulation direction Brier score | 0.4509 | 0.3983 | 0.3952 |
| P10–P90 coverage | 68% | 80.4% | 85.9% |

Validation covers 7 September to 4 October 2026. The local test covers the challenge test window (5 October to 1 November 2026 in Finnish local time): 673 issue times, each with 40 steps, for 26,920 rows per metric type. The team ran the challenge scorer against actuals reconstructed from the input history. These are aggregate scores from the original project. The local test column is an after-the-fact benchmark and does not report leaderboard results.

Published system state and known delivery-period forecasts were the main inputs. The challenge's validation period also informed model selection, so these validation improvements are not an independent estimate of performance on a new market regime. A sharp unexpected outage can still exceed the predicted range.
