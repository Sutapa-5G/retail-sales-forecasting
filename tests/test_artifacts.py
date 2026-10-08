"""Integration checks on the saved model, forecast and processed data (skipped if not generated)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src import data_preprocessing as dp
from src import evaluate as ev
from src import feature_engineering as fe
from src import forecast as fc

needs_model = pytest.mark.skipif(not config.MODEL_FILE.exists(), reason="run `python -m src.train` first")
needs_data = pytest.mark.skipif(not config.DAILY_FILE.exists(), reason="run the data pipeline first")


@needs_data
def test_processed_daily_table_is_clean_and_continuous():
    daily = dp.load_daily_metrics()
    assert (daily["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert daily["date"].is_unique
    assert (daily["revenue"] >= 0).all()
    assert str(daily["date"].iloc[0].date()) == config.FROZEN_SPLIT["series_start"]
    assert str(daily["date"].iloc[-1].date()) == config.FROZEN_SPLIT["series_end"]
    assert ((daily["revenue"] > 0) == (daily["is_trading_day"] == 1)).all()


@needs_data
def test_cleaning_log_reconciles_with_raw_rows_and_daily_revenue():
    log = pd.read_csv(config.CLEANING_LOG_FILE)
    summary = json.loads(config.SUMMARY_FILE.read_text())
    assert log["rows"].sum() == summary["raw_rows"]
    included_value = log.loc[log["status"] == "included", "line_value_gbp"].iloc[0]
    assert included_value == pytest.approx(summary["total_revenue_gbp"], abs=5.0)   # daily values are rounded to pence


@needs_data
def test_country_tables_reconcile_with_daily_revenue():
    daily = dp.load_daily_metrics()
    country = pd.read_csv(config.COUNTRY_DAILY_FILE)
    assert country["revenue"].sum() == pytest.approx(daily["revenue"].sum(), abs=20.0)


@needs_model
def test_saved_model_loads_and_metadata_matches_current_features():
    model, meta = fc.load_artifacts()
    assert meta["feature_columns"] == fe.FEATURE_COLUMNS
    assert meta["forecast_horizon_days"] == config.FORECAST_HORIZON
    assert hasattr(model, "predict")
    for key in ("model_name", "hyperparameters", "saved_model_training_period", "versions",
                "validation_metrics_mean_of_folds", "test_metrics_pool_trained_model", "prediction_interval"):
        assert key in meta


@needs_model
@needs_data
def test_forecast_artifact_is_reproduced_by_the_saved_model():
    model, meta = fc.load_artifacts()
    history = dp.load_revenue_series()
    again = fc.forecast_with_artifacts(model, meta, history, config.FORECAST_HORIZON)
    saved = pd.read_csv(config.FORECAST_FILE, parse_dates=["date"]).set_index("date")
    assert len(saved) == config.FORECAST_HORIZON
    assert saved.index[0] == history.index[-1] + pd.Timedelta(days=1)
    assert np.allclose(again["forecast"].to_numpy(), saved["forecast"].to_numpy(), atol=0.01)
    again2 = fc.forecast_with_artifacts(model, meta, history, config.FORECAST_HORIZON)
    assert again["forecast"].equals(again2["forecast"])                    # deterministic


@needs_model
def test_saved_forecast_is_sane():
    saved = pd.read_csv(config.FORECAST_FILE, parse_dates=["date"]).set_index("date")
    assert (saved["forecast"] >= 0).all()
    assert (saved["lower"] <= saved["forecast"] + 1e-6).all()
    assert (saved["upper"] >= saved["forecast"] - 1e-6).all()
    closed = fe.known_closure_mask(saved.index)
    assert (saved.loc[closed, "forecast"] == 0).all()


@needs_model
def test_invalid_horizon_is_rejected_with_the_real_model():
    model, meta = fc.load_artifacts()
    history = dp.load_revenue_series()
    for bad in (0, config.FORECAST_HORIZON + 1):
        with pytest.raises(ValueError):
            fc.forecast_with_artifacts(model, meta, history, bad)


@needs_model
@needs_data
def test_reported_test_metrics_can_be_recomputed_from_saved_predictions():
    meta = json.loads(config.MODEL_METADATA_FILE.read_text())
    pred = pd.read_csv(config.TEST_PREDICTIONS_FILE)
    recomputed = ev.compute_metrics(pred["actual"], pred["forecast"])
    reported = meta["test_metrics_pool_trained_model"]["30-day recursive"]
    assert recomputed["mae"] == pytest.approx(reported["mae"], abs=0.05)
    assert recomputed["rmse"] == pytest.approx(reported["rmse"], abs=0.05)
    assert len(pred) == config.TEST_DAYS


@needs_model
def test_selection_recorded_in_comparison_file_is_the_best_validation_model():
    comp = pd.read_csv(config.MODEL_COMPARISON_FILE)
    meta = json.loads(config.MODEL_METADATA_FILE.read_text())
    val = comp[(comp["split"].str.startswith("validation")) & (comp["protocol"] == "30-day recursive")
               & (comp["kind"] == "ml")]
    assert val.loc[val["mae"].idxmin(), "model"] == meta["model_name"]
    assert comp.loc[comp["selected_final_model"] & comp["split"].str.startswith("validation"), "model"].nunique() == 1
    # the test split contains the selected model plus the (untuned) baselines only
    test_models = set(comp.loc[comp["split"].str.startswith("test") & (comp["kind"] == "ml"), "model"])
    assert test_models == {meta["model_name"]}
