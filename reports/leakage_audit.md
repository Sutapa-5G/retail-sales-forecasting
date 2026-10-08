# Leakage Audit

Time-series projects fail quietly when information from the future reaches the model. This audit lists every risk checked, whether it was **avoided**, how, and where the evidence lives. Test names refer to `tests/`; notebook checks refer to `notebooks/03_feature_engineering.ipynb`.

## Summary

All 16 listed risks were **avoided**. Section B records one development iteration in full, and section C lists the residual caveats that remain (none of them is leakage, but they limit how far results should be trusted).

## A. Risk-by-risk audit

| # | Risk | Status | How it is avoided | Evidence |
|---|---|---|---|---|
| 1 | Random train/test splitting | **Avoided** | The final test set is the last 30 days; validation uses expanding windows where training always ends before validation starts. No shuffling, `train_test_split` or `KFold` is used anywhere. | `test_chronological_split_is_ordered_and_disjoint`, `test_expanding_window_folds_are_chronological`, `test_every_validation_fold_trains_strictly_before_it_validates`, `test_frozen_split_matches_computed_folds_on_the_real_date_range` |
| 2 | Future observations entering training features | **Avoided** | Features of day t use calendar facts about t and revenue strictly before t. | Notebook 03 check "changing day t and later never changes features of day t or earlier"; `test_features_at_t_do_not_change_when_t_or_the_future_changes` |
| 3 | Incorrect lag calculation | **Avoided** | `lag_k = series.shift(k)` with k in {1, 7, 14, 28}; never k = 0. | `test_lag_feature_uses_value_k_days_earlier` (every k); notebook 03 lag check on 25 sampled days |
| 4 | Incorrect rolling calculation | **Avoided** | `series.shift(1).rolling(w).mean()`: the window ends the day before the prediction date, so today's target is excluded. | `test_rolling_mean_covers_previous_w_days_and_excludes_today` (every w); notebook 03 rolling checks |
| 5 | Test data influencing feature engineering | **Avoided** | Feature construction is stateless (nothing is fitted). Calendar rules (holidays, 24 Dec - 3 Jan closure, Easter Sunday) were verified on the pool only; the test window contains 0 flagged days, so those rules cannot affect test predictions. | `test_features_never_depend_on_data_after_the_series_end`; notebook 03 section 1 (pool only) |
| 6 | Test data influencing scaling / imputation | **Avoided** | No imputation exists anywhere (incomplete leading rows are dropped, not filled). The only scaler (Linear Regression) lives inside a pipeline fitted on training rows only. | `test_scaler_inside_linear_pipeline_is_fitted_on_training_rows_only` |
| 7 | Model fitting on data it is evaluated on | **Avoided** | Each validation fold fits on history up to its `train_end`; the test evaluation fits on the pool; only after evaluation is the chosen configuration refit on all data for the production forecast. | `test_fitting_uses_only_the_series_it_is_given`; `run_training` order of operations |
| 8 | Hyperparameter tuning using test data | **Avoided** | All 15 configurations are scored on the validation folds only; `run_validation` receives the pool and has no access to the test series. | `test_changing_only_the_test_period_cannot_change_validation_or_selection` (shifts the test period by 75,000 and shows validation results, selection and interval quantiles are identical) |
| 9 | Test-set usage | **Avoided** | The test set is scored once, for the selected model, in one call (`evaluate_on_test`). The four baselines are also scored on it for reference only (they have no fitted parameters and played no role in selection). The other candidate models are never scored on the test set. | `test_selection_recorded_in_comparison_file_is_the_best_validation_model` (only the selected model appears among ML rows on the test split) |
| 10 | Model selection on test performance | **Avoided** | Rule fixed in advance: lowest mean validation MAE under the 30-day recursive protocol. | `test_selection_rule_is_lowest_validation_recursive_mae_among_models`; `meta["selection_rule"]` |
| 11 | Forecast-horizon contamination | **Avoided** | Multi-step forecasts are recursive: later days use the model's own earlier forecasts as lags. The target day is appended as NaN so its own value cannot influence its features. | `test_recursion_feeds_its_own_forecasts_back_as_lags`, `test_first_step_features_equal_the_vectorized_features`, `test_forecast_never_reads_actual_future_values`; `test_test_forecast_does_not_use_test_actuals` |
| 12 | Missing-date handling | **Avoided** | The calendar is continuous; days without sales are explicit zeros. Nothing is interpolated, back-filled or forward-filled, which would use neighbouring (future) days. | `test_daily_aggregation_values_and_zero_filled_days`; non-continuous input is rejected (`test_non_daily_index_is_rejected`) |
| 13 | Outlier handling | **Avoided** | No threshold is estimated from data and nothing is winsorized or removed by size. The one outlier-related rule (same-day reversal) is a row-level rule that uses only events of the same calendar day. | `test_same_day_reversal_matches_one_to_one`, `test_cancellation_before_sale_is_not_a_reversal` |
| 14 | Aggregation | **Avoided** | Daily revenue is a plain sum over one calendar day's lines. The partial last day is excluded so that an artificial low day does not appear in the test window. | `test_daily_aggregation_values_and_zero_filled_days`, `test_partial_last_day_is_detected_and_excluded` |
| 15 | Feature availability at prediction time | **Avoided** | Every feature is either a calendar fact known in advance or a function of revenue strictly before the prediction date. No customer-, country-, cancellation- or price-based same-day information is used as a feature. | `meta["feature_columns"]`; feature catalogue in notebook 03 |
| 16 | Prediction intervals using test information | **Avoided** | Residual quantiles come from out-of-sample validation forecasts of the selected model only. The test period is used only to *report* coverage afterwards. | Same test as #8 (quantiles identical when the test period changes) |

## B. Development log (iteration transparency)

The workflow was run twice. Both runs are recorded because the second was triggered by something seen in the first.

**Iteration 1.** Selected model: **XGBoost (n=400, depth=5, lr=0.03)**; validation MAE £8,590.60 (best baseline £10,796.76); test MAE £8,674.13. The 30-day *future* forecast then predicted £7k - £32k per day for 25 December - 3 January, a window with zero sales in both observed years.

**Diagnosis.** In-sample the model fitted the closure rows, but in recursive forecasting the lag and rolling features at Christmas (still high from the autumn peak) look nothing like the 17 closure rows in training, so a tree model trained on so few rows cannot generalize "closed means zero".

**Iteration 2 change.** A deterministic calendar rule was added: forecasts are set to zero on UK holidays and 24 Dec - 3 Jan, inside the recursion, for the models **and the baselines** (so the comparison stays fair). The rule is justified by the pool alone: every one of the 31 non-Saturday zero-revenue days in the pool is a flagged day, and no flagged day had sales. Saturdays are excluded from the rule because one Saturday did trade.

**Was this test-set tuning?** No, for three reasons. (i) The trigger was the future forecast window, not test performance. (ii) The test window contains **0** flagged days, so the rule cannot alter test predictions: the test forecasts, one-step forecasts and baseline forecasts were compared before and after and were **identical** (maximum difference 0.0000). (iii) The same model was selected afterwards. Transparency note: the first run's test predictions had already been printed before the change was made, so this iteration is documented rather than hidden.

**After the change.** Selected model: **XGBoost (n=400, depth=5, lr=0.03)** (unchanged); validation MAE £8,490.07 (the small improvement comes from the one flagged day inside any validation window: the 29 August 2011 bank holiday in fold 2); test MAE £8,674.13 (unchanged). Only the lower interval bound moved, by at most £1.92, because it is built from validation residuals.

## C. Residual caveats (not leakage, but real)

1. **Selection optimism.** The winner was chosen among 15 configurations on the same four validation folds, so its validation score is slightly optimistic. The untouched test set is the unbiased estimate, but 30 days is a small sample (one November-December window).
2. **Four validation windows.** Differences between the top models are small relative to fold-to-fold variation; the model that wins on average does not win every fold.
3. **Calendar rules learned from two years.** The holiday and year-end closure rules are consistent with 100% of the pool, but two years is a short history. If the business changed its opening pattern, the rule would need revisiting.
4. **Descriptive analyses use all rows.** The data audit (notebook 01) and `insights.md` describe the whole period, including the test window. They inform cleaning rules and business reading, never model selection; notebooks 02-03 use the pool only.
5. **Partial-day rule.** It compares last-invoice *times* across all days (including test-period days). It uses timestamps, not revenue, and protects the test set from an artificial low day.
6. **Final refit.** The saved production model is refit on all data through 2011-12-08 with frozen hyperparameters. Its own accuracy cannot be measured on held-out data; the reported test metrics belong to the pool-trained version.
7. **Interval coverage.** The empirical 80% interval covered 83% of the 30 test days, which is a small-sample check, not a guarantee.
