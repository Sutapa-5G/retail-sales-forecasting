"""The Streamlit dashboard starts from saved artifacts and handles bad input gracefully."""
import datetime as dt

import pytest

from src import config

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = config.PROJECT_ROOT / "app" / "app.py"
READY = config.MODEL_FILE.exists() and config.DAILY_FILE.exists()
pytestmark = pytest.mark.skipif(not READY, reason="run the pipeline and `python -m src.train` first")


def _start():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_app_starts_without_errors_and_shows_all_sections():
    at = _start()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Executive Overview", "Historical Sales", "Sales Analysis",
                                          "Forecast", "Model Performance", "Business Insights"]
    labels = {m.label for m in at.metric}
    for expected in ("Total revenue (selected range)", "Average daily revenue", "Total transactions (invoices)",
                     "Unique customers (whole period)", "Forecast horizon", "Latest actual revenue"):
        assert expected in labels


def test_horizon_control_changes_the_forecast():
    at = _start()
    full = [m.value for m in at.metric if m.label.startswith("Forecast total")][0]
    at.sidebar.slider[0].set_value(7).run()
    assert not at.exception
    short = [m.value for m in at.metric if m.label.startswith("Forecast total")][0]
    assert short != full
    assert [m.value for m in at.metric if m.label == "Forecast horizon"] == ["7 days"]


def test_reversed_date_range_shows_a_warning_not_a_crash():
    at = _start()
    at.sidebar.date_input[0].set_value((dt.date(2011, 6, 1), dt.date(2011, 1, 1))).run()
    assert not at.exception
    assert any("start date is after the end date" in w.value for w in at.warning)


def test_incomplete_date_selection_is_handled():
    at = _start()
    at.sidebar.date_input[0].set_value((dt.date(2011, 6, 1),)).run()
    assert not at.exception
    assert any("both a start and an end date" in i.value for i in at.info)


def test_empty_country_selection_and_closed_day_range_do_not_crash():
    at = _start()
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception
    at.sidebar.date_input[0].set_value((dt.date(2011, 3, 5), dt.date(2011, 3, 5))).run()   # a closed Saturday
    assert not at.exception


def test_app_forecast_equals_the_saved_forecast_artifact():
    import pandas as pd
    at = _start()
    total = [m.value for m in at.metric if m.label.startswith("Forecast total")][0]
    saved = pd.read_csv(config.FORECAST_FILE)["forecast"].sum()
    assert total == f"£{saved:,.0f}"
