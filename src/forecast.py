"""Recursive multi-step forecasting, empirical prediction intervals, artifact loading.

Recursive strategy: to forecast day T+h the model needs lag/rolling features.
Those are built from the observed history plus the model's OWN earlier
forecasts for T+1..T+h-1. No actual future value is ever used.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src import feature_engineering as fe


def validate_history(history: pd.Series) -> None:
    """Raise a clear error if ``history`` cannot be used to forecast."""
    if not isinstance(history, pd.Series):
        raise TypeError("history must be a pandas Series indexed by date")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("history must have a DatetimeIndex")
    if len(history) < config.MAX_LOOKBACK:
        raise ValueError(f"history needs at least {config.MAX_LOOKBACK} days, got {len(history)}")
    if history.isna().any():
        raise ValueError("history contains missing values")
    if (history < 0).any():
        raise ValueError("history contains negative revenue")
    gaps = history.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError("history must have a continuous daily index")


def recursive_forecast(
    model,
    history: pd.Series,
    horizon: int,
    feature_columns=None,
    max_horizon: int = config.FORECAST_HORIZON,
    apply_closures: bool = True,
) -> pd.Series:
    """Forecast ``horizon`` days after the end of ``history``, one day at a time.

    Predictions are clipped at 0 because revenue cannot be negative. When
    ``apply_closures`` is True, days known in advance to be closed (UK holidays,
    24 Dec - 3 Jan) are set to 0 INSIDE the recursion, so later lag/rolling
    features see the zero just like they do in the historical data.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer number of days")
    if horizon < 1 or horizon > max_horizon:
        raise ValueError(f"horizon must be between 1 and {max_horizon} days, got {horizon}")
    validate_history(history)

    columns = list(feature_columns or fe.FEATURE_COLUMNS)
    window = history.astype(float).iloc[-config.MAX_LOOKBACK:].copy()
    future = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    closed = fe.known_closure_mask(future) if apply_closures else np.zeros(horizon, dtype=bool)

    predictions = []
    for step, day in enumerate(future):
        # The target day is appended as NaN: its own value cannot influence its features.
        placeholder = pd.Series([np.nan], index=pd.DatetimeIndex([day]))
        row = fe.build_features(pd.concat([window, placeholder])).iloc[[-1]][columns]
        if row.isna().any().any():
            raise ValueError("feature row contains missing values; history too short")
        value = max(float(np.asarray(model.predict(row))[0]), 0.0)
        if closed[step]:
            value = 0.0
        predictions.append(value)
        window = pd.concat([window.iloc[1:], pd.Series([value], index=pd.DatetimeIndex([day]))])
    return pd.Series(predictions, index=future, name="forecast")


# ----------------------------------------------------------------------------
# Empirical prediction intervals
# ----------------------------------------------------------------------------
def closed_calendar_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """Days known in advance to be (almost always) closed: Saturdays, holidays, year-end."""
    cal = fe.calendar_features(index)
    return ((cal["day_of_week"] == 5) | (cal["is_uk_holiday"] == 1) | (cal["is_year_end_closure"] == 1)).to_numpy()


def fit_residual_quantiles(actual, predicted, index, coverage: float = config.INTERVAL_COVERAGE) -> dict:
    """Residual quantiles (actual - forecast) separately for closed-calendar and open days.

    These come from out-of-sample validation forecasts only. The interval is an
    empirical approximation, not a model-based guarantee.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    closed = closed_calendar_flags(pd.DatetimeIndex(index))
    residual = actual - predicted
    lo, hi = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    out = {"coverage": coverage}
    for name, mask in (("closed", closed), ("open", ~closed)):
        values = residual[mask]
        if len(values) == 0:
            values = residual
        out[name] = {
            "lower": float(np.quantile(values, lo)),
            "upper": float(np.quantile(values, hi)),
            "n": int(mask.sum()),
        }
    return out


def add_intervals(forecast: pd.Series, quantiles: dict) -> pd.DataFrame:
    """Forecast with lower/upper bounds from the residual quantiles (bounds >= 0)."""
    closed = closed_calendar_flags(forecast.index)
    lower_shift = np.where(closed, quantiles["closed"]["lower"], quantiles["open"]["lower"])
    upper_shift = np.where(closed, quantiles["closed"]["upper"], quantiles["open"]["upper"])
    lower = np.maximum(forecast.to_numpy() + lower_shift, 0.0)
    upper = np.maximum(forecast.to_numpy() + upper_shift, lower)
    return pd.DataFrame({"forecast": forecast.to_numpy(), "lower": lower, "upper": upper},
                        index=forecast.index)


# ----------------------------------------------------------------------------
# Saved artifacts
# ----------------------------------------------------------------------------
def load_artifacts(model_path: Path = config.MODEL_FILE, metadata_path: Path = config.MODEL_METADATA_FILE):
    """Load the saved model and its metadata (never retrains)."""
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    for path in (model_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}. Run `python -m src.train` first.")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    if list(metadata["feature_columns"]) != fe.FEATURE_COLUMNS:
        raise ValueError("saved model was trained with different features than the current code")
    return model, metadata


def forecast_with_artifacts(model, metadata: dict, history: pd.Series, horizon: int) -> pd.DataFrame:
    """Forecast + empirical interval using a saved model and its metadata."""
    prediction = recursive_forecast(model, history, horizon, metadata["feature_columns"])
    return add_intervals(prediction, metadata["prediction_interval"]["residual_quantiles"])
