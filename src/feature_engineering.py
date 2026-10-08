"""Leakage-safe time-series features for daily revenue forecasting.

Rules enforced here:
* Calendar features depend only on the date itself (known in advance).
* Lag features use ``shift(k)`` with k >= 1: they only read values BEFORE the
  prediction date.
* Rolling features are computed on ``series.shift(1)``, so the window ends the
  day BEFORE the prediction date and never includes the current target.
* Nothing here is fitted on data (no scaling/imputation), so nothing can leak
  from validation/test periods into feature construction.
"""
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

CALENDAR_COLUMNS = [
    "day_of_week", "day_of_month", "month", "quarter", "week_of_year",
    "day_of_year", "is_weekend", "is_uk_holiday", "is_year_end_closure",
]
LAG_COLUMNS = [f"lag_{k}" for k in config.LAG_PERIODS]
ROLLING_COLUMNS = [f"rolling_mean_{w}" for w in config.ROLLING_WINDOWS]
FEATURE_COLUMNS = CALENDAR_COLUMNS + LAG_COLUMNS + ROLLING_COLUMNS


# ----------------------------------------------------------------------------
# UK bank holidays (deterministic calendar knowledge)
# ----------------------------------------------------------------------------
def easter_sunday(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) given weekday of a month; n = -1 means the last one."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def uk_holidays(years: Iterable[int]) -> set:
    """Days the business was closed for UK public holidays (outside the year-end window).

    Good Friday, Easter Sunday, Easter Monday, early May, spring and summer
    bank holidays, plus one-off holidays listed in config.ONE_OFF_HOLIDAYS.
    Easter Sunday is included because no sales were recorded on it in either
    observed year. Christmas/Boxing Day/New Year are covered by the separate
    year-end closure feature.
    """
    days = set()
    for year in set(years):
        easter = easter_sunday(year)
        days.add(easter - timedelta(days=2))            # Good Friday
        days.add(easter)                                # Easter Sunday
        days.add(easter + timedelta(days=1))            # Easter Monday
        days.add(_nth_weekday(year, 5, 0, 1))           # early May (first Monday)
        days.add(_nth_weekday(year, 5, 0, -1))          # spring (last Monday of May)
        days.add(_nth_weekday(year, 8, 0, -1))          # summer (last Monday of August)
    days.update(pd.Timestamp(d).date() for d in config.ONE_OFF_HOLIDAYS)
    return days


def is_year_end_closure(index: pd.DatetimeIndex) -> np.ndarray:
    """True from 24 Dec to 3 Jan (the business recorded no sales in this window)."""
    start_m, start_d = config.YEAR_END_CLOSURE_START
    end_m, end_d = config.YEAR_END_CLOSURE_END
    month, day = index.month, index.day
    return ((month == start_m) & (day >= start_d)) | ((month == end_m) & (day <= end_d))


# ----------------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------------
def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features that depend only on the date (all known in advance)."""
    index = pd.DatetimeIndex(index)
    holidays = uk_holidays(index.year.unique())
    frame = pd.DataFrame(index=index)
    frame["day_of_week"] = index.dayofweek
    frame["day_of_month"] = index.day
    frame["month"] = index.month
    frame["quarter"] = index.quarter
    frame["week_of_year"] = index.isocalendar().week.to_numpy().astype(int)
    frame["day_of_year"] = index.dayofyear
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
    frame["is_uk_holiday"] = np.array([d.date() in holidays for d in index], dtype=int)
    frame["is_year_end_closure"] = is_year_end_closure(index).astype(int)
    return frame


def _check_daily_index(series: pd.Series) -> None:
    index = series.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    if len(index) > 1:
        gaps = index.to_series().diff().dropna()
        if not (gaps == pd.Timedelta(days=1)).all():
            raise ValueError("series must have a continuous daily index (no gaps, no duplicates)")


def build_features(series: pd.Series) -> pd.DataFrame:
    """Feature matrix aligned with ``series.index``.

    The value of ``series`` on date t is NEVER used to build the features of
    date t: lags use shift(k>=1) and rolling means use shift(1).rolling(w).
    The first ``max(lags, windows)`` rows contain NaN because not enough past
    data exists; callers drop them with :func:`make_training_set`.
    """
    _check_daily_index(series)
    features = calendar_features(series.index)
    for k in config.LAG_PERIODS:
        features[f"lag_{k}"] = series.shift(k)
    past = series.shift(1)
    for w in config.ROLLING_WINDOWS:
        features[f"rolling_mean_{w}"] = past.rolling(window=w, min_periods=w).mean()
    return features[FEATURE_COLUMNS]


def known_closure_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Days known IN ADVANCE to be closed: UK holidays and the year-end closure.

    In the training pool, revenue was exactly 0 on every one of these days and
    no flagged day had sales. Saturdays are NOT included because one Saturday
    did trade, so the model is left to handle them.
    """
    cal = calendar_features(index)
    return ((cal["is_uk_holiday"] == 1) | (cal["is_year_end_closure"] == 1)).to_numpy()


def make_training_set(series: pd.Series):
    """(X, y) with incomplete leading rows removed. Uses only the given series."""
    features = build_features(series)
    valid = features.notna().all(axis=1)
    return features.loc[valid], series.loc[valid]
