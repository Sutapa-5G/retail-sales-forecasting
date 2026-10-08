"""Metrics, chronological splitting, validation folds and baselines."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_series
from src import config
from src import evaluate as ev


# ---------------------------------------------------------------- metrics
def test_mae_rmse_known_values():
    y, f = [10, 20, 30], [12, 18, 36]
    assert ev.mae(y, f) == pytest.approx((2 + 2 + 6) / 3)
    assert ev.rmse(y, f) == pytest.approx(np.sqrt((4 + 4 + 36) / 3))


def test_wape_known_value():
    assert ev.wape([100, 100], [90, 130]) == pytest.approx(40 / 200 * 100)
    assert np.isnan(ev.wape([0, 0], [1, 1]))


def test_smape_ignores_closed_days():
    # day 1: actual 100 vs 50 -> 2*50/150 = 0.6667 ; closed day (actual 0) must be ignored
    assert ev.smape_trading_days([100, 0], [50, 999]) == pytest.approx(66.6667, abs=1e-3)
    assert np.isnan(ev.smape_trading_days([0, 0], [1, 2]))


def test_r2_perfect_and_constant_actuals():
    assert ev.r_squared([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert np.isnan(ev.r_squared([5, 5, 5], [4, 5, 6]))


def test_metric_input_validation():
    with pytest.raises(ValueError):
        ev.mae([1, 2], [1])
    with pytest.raises(ValueError):
        ev.mae([], [])


# ---------------------------------------------------------------- splitting
def test_chronological_split_is_ordered_and_disjoint():
    s = make_series(200)
    pool, test = ev.chronological_split(s, 30)
    assert len(test) == 30 and len(pool) == 170
    assert pool.index.max() < test.index.min()
    assert pool.index.union(test.index).equals(s.index)


def test_chronological_split_rejects_bad_sizes():
    s = make_series(50)
    for bad in (0, 50, 100):
        with pytest.raises(ValueError):
            ev.chronological_split(s, bad)


def test_expanding_window_folds_are_chronological():
    idx = pd.date_range("2010-01-01", periods=400, freq="D")
    folds = ev.expanding_window_folds(idx, n_folds=4, window=30)
    assert len(folds) == 4
    for f in folds:
        assert f.train_end < f.val_start <= f.val_end
        assert (f.val_end - f.val_start).days + 1 == 30
    for a, b in zip(folds[:-1], folds[1:]):
        assert b.train_end > a.train_end          # training window expands
        assert b.val_start > a.val_end            # validation windows do not overlap
    assert folds[-1].val_end == idx[-1]           # last fold ends at the end of the pool


def test_folds_reject_insufficient_history():
    idx = pd.date_range("2010-01-01", periods=150, freq="D")
    with pytest.raises(ValueError, match="not enough history"):
        ev.expanding_window_folds(idx, n_folds=4, window=30)


def test_frozen_split_matches_computed_folds_on_the_real_date_range():
    pool_index = pd.date_range(config.FROZEN_SPLIT["series_start"], config.FROZEN_SPLIT["pool_end"], freq="D")
    folds = ev.expanding_window_folds(pool_index)
    computed = [{"fold": f.fold, "train_end": str(f.train_end.date()), "val_start": str(f.val_start.date()),
                 "val_end": str(f.val_end.date())} for f in folds]
    assert computed == config.FROZEN_SPLIT["validation_folds"]
    full_index = pd.date_range(config.FROZEN_SPLIT["series_start"], config.FROZEN_SPLIT["series_end"], freq="D")
    pool, test = ev.chronological_split(pd.Series(0.0, index=full_index))
    assert str(test.index[0].date()) == config.FROZEN_SPLIT["test_start"]
    assert str(pool.index[-1].date()) == config.FROZEN_SPLIT["pool_end"]


# ---------------------------------------------------------------- baselines
def _history():
    idx = pd.date_range("2011-01-03", periods=70, freq="D")           # starts on a Monday
    return pd.Series(np.arange(1, 71, dtype=float) * 100, index=idx)


def test_naive_and_moving_average_baselines():
    h = _history()
    target = h.index[-1] + pd.Timedelta(days=1)
    assert ev.baseline_forecast("naive", h, target) == h.iloc[-1]
    assert ev.baseline_forecast("moving_average_7", h, target) == pytest.approx(h.iloc[-7:].mean())


def test_seasonal_naive_one_day_ahead_is_same_weekday_last_week():
    h = _history()
    target = h.index[-1] + pd.Timedelta(days=1)
    assert ev.baseline_forecast("seasonal_naive", h, target) == h.loc[target - pd.Timedelta(days=7)]


def test_seasonal_naive_reuses_latest_observed_same_weekday_for_longer_horizons():
    h = _history()
    target = h.index[-1] + pd.Timedelta(days=10)                      # needs 14 days back
    assert ev.baseline_forecast("seasonal_naive", h, target) == h.loc[target - pd.Timedelta(days=14)]
    assert (target - pd.Timedelta(days=14)) <= h.index[-1]            # never uses unobserved data


def test_seasonal_average_uses_four_same_weekdays():
    h = _history()
    target = h.index[-1] + pd.Timedelta(days=1)
    expected = np.mean([h.loc[target - pd.Timedelta(days=7 * k)] for k in (1, 2, 3, 4)])
    assert ev.baseline_forecast("seasonal_average_4w", h, target) == pytest.approx(expected)


def test_baseline_rejects_non_future_target_and_unknown_name():
    h = _history()
    with pytest.raises(ValueError):
        ev.baseline_forecast("naive", h, h.index[-1])
    with pytest.raises(ValueError):
        ev.baseline_forecast("magic", h, h.index[-1] + pd.Timedelta(days=1))


def test_baseline_window_applies_the_same_closure_rule_as_models():
    idx = pd.date_range("2011-11-01", periods=60, freq="D")
    h = pd.Series(1000.0, index=idx)                                   # last day 2011-12-30
    f = ev.baseline_window_forecast("naive", h.loc[:"2011-12-20"], 10)  # 21-30 Dec
    closed = f.loc["2011-12-24":]
    assert (closed == 0).all() and (f.loc[:"2011-12-23"] == 1000).all()
    unconstrained = ev.baseline_window_forecast("naive", h.loc[:"2011-12-20"], 10, apply_closures=False)
    assert (unconstrained == 1000).all()


def test_one_step_baseline_never_uses_the_target_day():
    s = make_series(100)
    dates = s.index[60:65]
    altered = s.copy()
    altered.loc[dates] = 1e9
    a = ev.baseline_one_step("seasonal_average_4w", s, dates)
    b = ev.baseline_one_step("seasonal_average_4w", altered, dates[:1])
    assert a.iloc[0] == pytest.approx(b.iloc[0])   # first day only uses the untouched past
