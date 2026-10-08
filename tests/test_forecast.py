"""Recursive forecasting, input validation and prediction intervals."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_series
from src import config
from src import feature_engineering as fe
from src import forecast as fc


class ConstantModel:
    def __init__(self, value):
        self.value, self.rows = value, []

    def predict(self, X):
        self.rows.append(X.copy())
        return np.full(len(X), self.value)


def test_forecast_length_and_dates_follow_history():
    s = make_series(120)
    out = fc.recursive_forecast(ConstantModel(1000.0), s, 14)
    assert len(out) == 14
    assert out.index[0] == s.index[-1] + pd.Timedelta(days=1)
    assert (out.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_recursion_feeds_its_own_forecasts_back_as_lags():
    s = make_series(120)
    model = ConstantModel(777.0)
    fc.recursive_forecast(model, s, 3, apply_closures=False)
    assert model.rows[1]["lag_1"].iloc[0] == 777.0                       # step 2 uses step-1 forecast
    first_day = s.index[-1] + pd.Timedelta(days=1)
    assert model.rows[0]["lag_1"].iloc[0] == s.iloc[-1]                  # step 1 uses the last actual


def test_first_step_features_equal_the_vectorized_features():
    s = make_series(150)
    hist = s.iloc[:120]
    model = ConstantModel(1.0)
    fc.recursive_forecast(model, hist, 1)
    target = hist.index[-1] + pd.Timedelta(days=1)
    vectorized = fe.build_features(s).loc[target]                        # built from the TRUE series
    assert np.allclose(model.rows[0].iloc[0].to_numpy(float), vectorized.to_numpy(float))


def test_forecast_never_reads_actual_future_values():
    s = make_series(150)
    hist = s.iloc[:120]
    a = fc.recursive_forecast(ConstantModel(500.0), hist, 10)
    b = fc.recursive_forecast(ConstantModel(500.0), s.iloc[:120].copy(), 10)
    assert a.equals(b)                                                   # nothing but history matters


def test_predictions_are_clipped_at_zero():
    out = fc.recursive_forecast(ConstantModel(-5000.0), make_series(100), 7)
    assert (out >= 0).all()


def test_known_closure_days_are_forced_to_zero_inside_the_recursion():
    idx = pd.date_range("2011-09-01", periods=110, freq="D")             # ends 2011-12-19
    s = pd.Series(30000.0, index=idx)
    model = ConstantModel(30000.0)
    out = fc.recursive_forecast(model, s, 14)                            # 20 Dec - 2 Jan
    assert (out.loc["2011-12-24":] == 0).all()
    assert (out.loc[:"2011-12-23"] == 30000.0).all()
    # lag_1 on the first day AFTER the closure (4 Jan is outside the horizon) is covered by the next test
    on_closure = model.rows[5]                                           # 25 Dec
    assert on_closure["lag_1"].iloc[0] == 0.0                            # 24 Dec forecast of 0 fed back


@pytest.mark.parametrize("bad", [0, -3, 31, 1000])
def test_invalid_horizon_values_raise(bad):
    with pytest.raises(ValueError):
        fc.recursive_forecast(ConstantModel(1.0), make_series(100), bad)


@pytest.mark.parametrize("bad", ["7", 7.5, None, True])
def test_non_integer_horizon_raises(bad):
    with pytest.raises(TypeError):
        fc.recursive_forecast(ConstantModel(1.0), make_series(100), bad)


def test_invalid_history_is_rejected():
    s = make_series(100)
    with pytest.raises(ValueError, match="at least"):
        fc.recursive_forecast(ConstantModel(1.0), s.iloc[:10], 5)
    with_nan = s.copy(); with_nan.iloc[50] = np.nan
    with pytest.raises(ValueError, match="missing"):
        fc.recursive_forecast(ConstantModel(1.0), with_nan, 5)
    negative = s.copy(); negative.iloc[50] = -1
    with pytest.raises(ValueError, match="negative"):
        fc.recursive_forecast(ConstantModel(1.0), negative, 5)
    gappy = s.drop(s.index[40])
    with pytest.raises(ValueError, match="continuous"):
        fc.recursive_forecast(ConstantModel(1.0), gappy, 5)
    with pytest.raises(TypeError):
        fc.recursive_forecast(ConstantModel(1.0), s.to_numpy(), 5)


# ---------------------------------------------------------------- intervals
def test_residual_quantiles_and_intervals_are_ordered_and_non_negative():
    idx = pd.date_range("2011-06-01", periods=60, freq="D")
    rng = np.random.default_rng(1)
    pred = rng.uniform(10000, 40000, 60)
    actual = pred + rng.normal(0, 5000, 60)
    q = fc.fit_residual_quantiles(actual, pred, idx, coverage=0.8)
    assert q["open"]["lower"] < 0 < q["open"]["upper"]
    forecast = pd.Series(rng.uniform(0, 40000, 20), index=pd.date_range("2011-12-01", periods=20))
    bands = fc.add_intervals(forecast, q)
    assert (bands["lower"] >= 0).all()
    assert (bands["lower"] <= bands["forecast"]).all()
    assert (bands["upper"] >= bands["lower"]).all()


def test_interval_coverage_is_close_to_nominal_on_the_fitting_residuals():
    idx = pd.date_range("2010-01-04", periods=400, freq="D")
    rng = np.random.default_rng(2)
    pred = np.full(400, 20000.0)
    actual = pred + rng.normal(0, 3000, 400)
    q = fc.fit_residual_quantiles(actual, pred, idx, coverage=0.8)
    bands = fc.add_intervals(pd.Series(pred, index=idx), q)
    covered = ((actual >= bands["lower"]) & (actual <= bands["upper"])).mean()
    assert covered == pytest.approx(0.8, abs=0.1)


def test_invalid_coverage_raises():
    with pytest.raises(ValueError):
        fc.fit_residual_quantiles([1, 2], [1, 2], pd.date_range("2011-01-01", periods=2), coverage=1.5)


def test_missing_artifacts_raise_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        fc.load_artifacts(tmp_path / "m.joblib", tmp_path / "m.json")
