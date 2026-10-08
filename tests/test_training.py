"""Training workflow: model selection uses validation only; the test set cannot leak in."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_series
from src import config
from src import evaluate as ev
from src import feature_engineering as fe
from src import train as tr

SMALL_SPECS = [
    tr.ModelSpec("Linear Regression", "linear"),
    tr.ModelSpec(
        "RF small",
        "random_forest",
        {"n_estimators": 15, "max_depth": 4, "min_samples_leaf": 3},
    ),
]


def _run(series):
    return tr.run_training(
        series,
        save=False,
        verbose=False,
        specs=SMALL_SPECS,
        verify_split=False,
        n_folds=2,
        window=14,
    )


def test_changing_only_the_test_period_cannot_change_validation_or_selection():
    base = make_series(320)
    shifted = make_series(320, test_shift=75000.0)

    assert base.iloc[:-30].equals(shifted.iloc[:-30])

    a, b = _run(base), _run(shifted)

    pd.testing.assert_frame_equal(a["summary"], b["summary"])
    pd.testing.assert_frame_equal(a["fold_metrics"], b["fold_metrics"])
    assert a["spec"].name == b["spec"].name

    a_quantiles = a["metadata"]["prediction_interval"]["residual_quantiles"]
    b_quantiles = b["metadata"]["prediction_interval"]["residual_quantiles"]

    assert a_quantiles["coverage"] == b_quantiles["coverage"]

    for status in ("closed", "open"):
        assert a_quantiles[status]["n"] == b_quantiles[status]["n"]
        assert a_quantiles[status]["lower"] == pytest.approx(
            b_quantiles[status]["lower"], abs=1e-9
        )
        assert a_quantiles[status]["upper"] == pytest.approx(
            b_quantiles[status]["upper"], abs=1e-9
        )

    # The test score DOES react to the test data.
    assert a["test_result"]["metrics"] != b["test_result"]["metrics"]


def test_test_forecast_does_not_use_test_actuals():
    base = make_series(320)
    shifted = make_series(320, test_shift=75000.0)

    a, b = _run(base), _run(shifted)

    pd.testing.assert_series_equal(
        a["test_result"]["predictions"]["forecast"],
        b["test_result"]["predictions"]["forecast"],
    )


def test_selection_rule_is_lowest_validation_recursive_mae_among_models():
    rows = []

    for name, family, kind, proto, mae in [
        ("Baseline", "baseline", "baseline", tr.PROTOCOL_RECURSIVE, 1.0),
        ("Model A", "linear", "ml", tr.PROTOCOL_RECURSIVE, 50.0),
        ("Model B", "random_forest", "ml", tr.PROTOCOL_RECURSIVE, 40.0),
        ("Model A", "linear", "ml", tr.PROTOCOL_ONE_STEP, 10.0),
        ("Model B", "random_forest", "ml", tr.PROTOCOL_ONE_STEP, 90.0),
    ]:
        rows.append(
            {
                "model": name,
                "family": family,
                "kind": kind,
                "protocol": proto,
                "params": "{}",
                "mae": mae,
                "rmse": mae,
                "smape_trading_days": 1.0,
                "wape": 1.0,
                "r2": 0.0,
                "n_folds": 4,
            }
        )

    specs = [
        tr.ModelSpec("Model A", "linear"),
        tr.ModelSpec("Model B", "random_forest"),
    ]

    chosen, per_family = tr.select_final_model(pd.DataFrame(rows), specs)

    assert chosen.name == "Model B"
    assert set(per_family["family"]) == {"linear", "random_forest"}


def test_fitting_uses_only_the_series_it_is_given():
    s = make_series(260)

    spec = tr.ModelSpec(
        "RF small",
        "random_forest",
        {"n_estimators": 15, "max_depth": 4, "min_samples_leaf": 3},
    )

    train_part = s.iloc[:200]

    future_changed = s.copy()
    future_changed.iloc[200:] = 1e7

    m1 = tr.fit_model(spec, train_part)
    m2 = tr.fit_model(spec, future_changed.iloc[:200])

    X = fe.build_features(s).iloc[150:200][fe.FEATURE_COLUMNS]

    assert np.allclose(m1.predict(X), m2.predict(X))


def test_scaler_inside_linear_pipeline_is_fitted_on_training_rows_only():
    s = make_series(300)

    model = tr.fit_model(
        tr.ModelSpec("Linear Regression", "linear"),
        s.iloc[:200],
    )

    scaler = model.named_steps["prep"].named_transformers_["numeric"]

    X_train, _ = fe.make_training_set(s.iloc[:200])
    assert scaler.mean_[1] == pytest.approx(X_train["lag_1"].mean())

    X_all, _ = fe.make_training_set(s)
    assert scaler.mean_[1] != pytest.approx(X_all["lag_1"].mean())


def test_frozen_split_check_refuses_mismatching_data():
    idx = pd.date_range("2010-01-04", periods=400, freq="D")
    s = pd.Series(1.0, index=idx)

    pool, test = ev.chronological_split(s)
    folds = ev.expanding_window_folds(pool.index)

    with pytest.raises(ValueError, match="Frozen split mismatch"):
        tr.verify_frozen_split(s, pool, test, folds)


def test_candidate_grid_is_small_and_explainable():
    specs = tr.candidate_specs()

    assert len(specs) == 15
    assert {s.family for s in specs} == {
        "linear",
        "random_forest",
        "xgboost",
    }
    assert len({s.name for s in specs}) == len(specs)


def test_every_validation_fold_trains_strictly_before_it_validates():
    s = make_series(320)

    result = _run(s)

    for fold in result["folds"]:
        assert fold.train_end < fold.val_start

    val = result["val_predictions"]

    assert val.groupby("fold")["date"].min().is_monotonic_increasing