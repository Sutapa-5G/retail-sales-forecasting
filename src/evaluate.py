"""Metrics, baseline forecasters and chronological validation splits."""
from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src import config
from src import feature_engineering as fe

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _as_arrays(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {f.shape}")
    if a.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    return a, f


def mae(y_true, y_pred) -> float:
    """Mean absolute error in GBP: the average daily miss."""
    a, f = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - f)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error in GBP: like MAE but punishes large misses more."""
    a, f = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error (%): total absolute miss / total actual revenue."""
    a, f = _as_arrays(y_true, y_pred)
    total = np.sum(np.abs(a))
    return float("nan") if total == 0 else float(np.sum(np.abs(a - f)) / total * 100)


def smape_trading_days(y_true, y_pred) -> float:
    """Symmetric MAPE (%) on days with actual revenue > 0.

    Plain MAPE is undefined on closed days (actual = 0) and sMAPE would score
    every tiny forecast on a closed day as a 200% error, so closed days are
    excluded here. They are still fully counted in MAE, RMSE and WAPE.
    """
    a, f = _as_arrays(y_true, y_pred)
    mask = a > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2 * np.abs(a[mask] - f[mask]) / (np.abs(a[mask]) + np.abs(f[mask]))) * 100)


def r_squared(y_true, y_pred) -> float:
    """R-squared (supplementary only; not used for model selection)."""
    a, f = _as_arrays(y_true, y_pred)
    if a.size < 2 or np.allclose(a, a[0]):
        return float("nan")
    return float(r2_score(a, f))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape_trading_days": smape_trading_days(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# Chronological splitting
# ----------------------------------------------------------------------------
def chronological_split(series: pd.Series, test_days: int = config.TEST_DAYS):
    """Split into (pool, test): the test set is the LAST ``test_days`` days.

    ``pool`` (everything before the test window) is the only data that may be
    used for feature exploration, model fitting and hyper-parameter selection.
    """
    if test_days < 1 or test_days >= len(series):
        raise ValueError("test_days must be between 1 and len(series) - 1")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series index must be sorted in time order")
    return series.iloc[:-test_days], series.iloc[-test_days:]


@dataclass(frozen=True)
class Fold:
    fold: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def expanding_window_folds(
    pool_index: pd.DatetimeIndex,
    n_folds: int = config.N_CV_FOLDS,
    window: int = config.CV_WINDOW_DAYS,
    min_train_days: int = 180,
) -> list:
    """Expanding-window folds counted back from the end of the pool.

    Each fold trains on everything up to ``train_end`` and validates on the
    next ``window`` days, so validation is always strictly after training.
    """
    pool_end = pool_index[-1]
    folds = []
    for k in range(n_folds):
        val_end = pool_end - pd.Timedelta(days=window * (n_folds - 1 - k))
        val_start = val_end - pd.Timedelta(days=window - 1)
        train_end = val_start - pd.Timedelta(days=1)
        if (train_end - pool_index[0]).days + 1 < min_train_days:
            raise ValueError("not enough history for the requested number of folds")
        folds.append(Fold(k + 1, train_end, val_start, val_end))
    return folds


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
BASELINE_LABELS = {
    "naive": "Naive (last observed day)",
    "seasonal_naive": "Seasonal naive (same weekday last week)",
    "moving_average_7": "Moving average (last 7 days)",
    "seasonal_average_4w": "Seasonal average (same weekday, last 4 weeks)",
}
BASELINE_NAMES = list(BASELINE_LABELS)


def baseline_forecast(name: str, history: pd.Series, target_date) -> float:
    """Forecast one date from ``history`` (which ends at the forecast origin).

    With history ending the day before ``target_date`` these are the classic
    1-day-ahead baselines (previous day, same day last week, ...). For dates
    further ahead the same rule is applied from the origin, e.g. seasonal
    naive reuses the most recent same weekday that is already observed.
    """
    target_date = pd.Timestamp(target_date)
    horizon = (target_date - history.index[-1]).days
    if horizon < 1:
        raise ValueError("target_date must be after the end of history")
    if name == "naive":
        return float(history.iloc[-1])
    if name == "moving_average_7":
        return float(history.iloc[-7:].mean())
    weeks_back = ceil(horizon / 7)
    if name == "seasonal_naive":
        return float(history.loc[target_date - pd.Timedelta(days=7 * weeks_back)])
    if name == "seasonal_average_4w":
        dates = [target_date - pd.Timedelta(days=7 * (weeks_back + j)) for j in range(4)]
        return float(history.loc[dates].mean())
    raise ValueError(f"unknown baseline: {name}")


def _apply_closures(values: pd.Series, apply_closures: bool) -> pd.Series:
    """Set known closure days (holidays, year-end) to 0, same rule as for the models."""
    if apply_closures:
        values = values.mask(fe.known_closure_mask(values.index), 0.0)
    return values


def baseline_window_forecast(name: str, history: pd.Series, horizon: int,
                             apply_closures: bool = True) -> pd.Series:
    """Multi-step baseline: forecast ``horizon`` days from the end of ``history``."""
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    out = pd.Series([baseline_forecast(name, history, d) for d in dates], index=dates, name=name)
    return _apply_closures(out, apply_closures)


def baseline_one_step(name: str, series: pd.Series, dates, apply_closures: bool = True) -> pd.Series:
    """1-day-ahead baseline for each date, using only data before that date."""
    values = [baseline_forecast(name, series.loc[: d - pd.Timedelta(days=1)], d) for d in dates]
    out = pd.Series(values, index=pd.DatetimeIndex(dates), name=name)
    return _apply_closures(out, apply_closures)
