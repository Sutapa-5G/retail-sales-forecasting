"""Calendar knowledge, lag/rolling correctness and absence of future leakage."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import make_series
from src import config
from src import feature_engineering as fe


def test_easter_sunday_matches_observed_dates():
    assert fe.easter_sunday(2010) == date(2010, 4, 4)
    assert fe.easter_sunday(2011) == date(2011, 4, 24)
    assert fe.easter_sunday(2012) == date(2012, 4, 8)


def test_uk_holidays_2011():
    days = fe.uk_holidays([2011])
    for d in [date(2011, 4, 22), date(2011, 4, 24), date(2011, 4, 25), date(2011, 4, 29),
              date(2011, 5, 2), date(2011, 5, 30), date(2011, 8, 29)]:
        assert d in days
    assert date(2011, 6, 1) not in days


def test_year_end_closure_window():
    idx = pd.to_datetime(["2010-12-23", "2010-12-24", "2010-12-31", "2011-01-03", "2011-01-04"])
    assert fe.is_year_end_closure(idx).tolist() == [False, True, True, True, False]


def test_known_closure_mask_excludes_saturdays():
    idx = pd.to_datetime(["2011-11-12",   # ordinary Saturday
                          "2011-04-25",   # Easter Monday
                          "2011-12-27"])  # year-end closure
    assert fe.known_closure_mask(idx).tolist() == [False, True, True]


def test_calendar_features_values():
    cal = fe.calendar_features(pd.DatetimeIndex(["2011-11-12"]))   # a Saturday
    row = cal.iloc[0]
    assert row["day_of_week"] == 5 and row["is_weekend"] == 1
    assert row["month"] == 11 and row["quarter"] == 4 and row["day_of_month"] == 12


def test_feature_columns_are_complete_and_ordered():
    X = fe.build_features(make_series(60))
    assert list(X.columns) == fe.FEATURE_COLUMNS
    for k in config.LAG_PERIODS:
        assert f"lag_{k}" in X.columns
    for w in config.ROLLING_WINDOWS:
        assert f"rolling_mean_{w}" in X.columns


@pytest.mark.parametrize("k", config.LAG_PERIODS)
def test_lag_feature_uses_value_k_days_earlier(k):
    s = make_series(120)
    X = fe.build_features(s)
    t = s.index[60]
    assert X.loc[t, f"lag_{k}"] == s.loc[t - pd.Timedelta(days=k)]


@pytest.mark.parametrize("w", config.ROLLING_WINDOWS)
def test_rolling_mean_covers_previous_w_days_and_excludes_today(w):
    s = make_series(120)
    X = fe.build_features(s)
    t = s.index[70]
    expected = s.loc[t - pd.Timedelta(days=w): t - pd.Timedelta(days=1)].mean()
    assert X.loc[t, f"rolling_mean_{w}"] == pytest.approx(expected)
    wrong_with_today = s.loc[t - pd.Timedelta(days=w - 1): t].mean()
    assert X.loc[t, f"rolling_mean_{w}"] != pytest.approx(wrong_with_today)


def test_features_at_t_do_not_change_when_t_or_the_future_changes():
    s = make_series(150)
    t = s.index[100]
    altered = s.copy()
    altered.loc[t:] = altered.loc[t:] * 9 + 12345.0           # change t and everything after
    a, b = fe.build_features(s), fe.build_features(altered)
    assert np.allclose(a.loc[t].to_numpy(float), b.loc[t].to_numpy(float))
    assert a.loc[:t].equals(b.loc[:t])


def test_features_do_change_when_the_past_changes():
    s = make_series(150)
    t = s.index[100]
    altered = s.copy()
    altered.loc[t - pd.Timedelta(days=1)] += 5000.0
    assert fe.build_features(altered).loc[t, "lag_1"] != fe.build_features(s).loc[t, "lag_1"]


def test_features_never_depend_on_data_after_the_series_end():
    s = make_series(150)
    short = s.iloc[:110]
    assert fe.build_features(s).loc[short.index].equals(fe.build_features(short))


def test_make_training_set_drops_rows_without_enough_history():
    s = make_series(100)
    X, y = fe.make_training_set(s)
    assert X.notna().all().all()
    assert X.index[0] == s.index[0] + pd.Timedelta(days=config.MAX_LOOKBACK)
    assert len(X) == len(y) == len(s) - config.MAX_LOOKBACK


def test_non_daily_index_is_rejected():
    s = make_series(60).iloc[::2]
    with pytest.raises(ValueError, match="continuous daily"):
        fe.build_features(s)
