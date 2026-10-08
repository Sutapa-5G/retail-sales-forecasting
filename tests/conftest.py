"""Shared fixtures: small synthetic data only (the real dataset is never needed for unit tests)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_transactions(rows):
    """Build a standardized transactions frame from tuples.

    (invoice_no, stock_code, quantity, unit_price, "YYYY-MM-DD HH:MM", customer_id, country)
    """
    frame = pd.DataFrame(rows, columns=["invoice_no", "stock_code", "quantity", "unit_price",
                                        "invoice_date", "customer_id", "country"])
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    frame["description"] = "ITEM"
    frame["invoice_no"] = frame["invoice_no"].astype("string")
    frame["stock_code"] = frame["stock_code"].astype("string")
    frame["customer_id"] = frame["customer_id"].astype(float)
    frame["source_sheet"] = "Year 2009-2010"
    return frame


def make_series(n_days=300, start="2010-01-04", seed=0, test_shift=0.0):
    """Synthetic daily revenue: weekly pattern, mild trend, noise, Saturdays closed."""
    index = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    weekday_level = np.array([30, 35, 32, 36, 28, 0, 17]) * 1000.0
    values = weekday_level[index.dayofweek] * (1 + index.dayofyear.to_numpy() / 900) + rng.normal(0, 1500, n_days)
    values = np.clip(values, 0, None)
    values[index.dayofweek.to_numpy() == 5] = 0.0
    series = pd.Series(values, index=index, name="revenue")
    if test_shift:
        series.iloc[-30:] = series.iloc[-30:] + test_shift
    return series


@pytest.fixture
def synthetic_series():
    return make_series()
