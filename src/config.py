"""Central, project-relative configuration. No machine-specific paths."""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "online_retail_II.xlsx"

DAILY_FILE = PROCESSED_DIR / "daily_metrics.csv"
COUNTRY_DAILY_FILE = PROCESSED_DIR / "country_daily.csv"
CLEANING_LOG_FILE = PROCESSED_DIR / "cleaning_log.csv"
FORECAST_FILE = PROCESSED_DIR / "forecast_30d.csv"
TEST_PREDICTIONS_FILE = PROCESSED_DIR / "test_predictions.csv"
VALIDATION_PREDICTIONS_FILE = PROCESSED_DIR / "validation_predictions.csv"

MODEL_FILE = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"

# ----------------------------------------------------------------------------
# Raw column handling. The real file headers are "Invoice", "Price" and
# "Customer ID"; the UCI web page documents InvoiceNo / UnitPrice / CustomerID.
# Both spellings are accepted and mapped to one standard set of names.
# ----------------------------------------------------------------------------
COLUMN_ALIASES = {
    "Invoice": "invoice_no", "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price", "UnitPrice": "unit_price",
    "Customer ID": "customer_id", "CustomerID": "customer_id",
    "Country": "country",
}
STANDARD_COLUMNS = [
    "invoice_no", "stock_code", "description", "quantity",
    "invoice_date", "unit_price", "customer_id", "country",
]

# ----------------------------------------------------------------------------
# Modeling configuration
# ----------------------------------------------------------------------------
DATE_COLUMN = "date"
TARGET_COLUMN = "revenue"          # daily gross revenue in GBP
FORECAST_HORIZON = 30              # days
RANDOM_SEED = 42

LAG_PERIODS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

# ----------------------------------------------------------------------------
# Cleaning rules (derived from the data audit, see reports/data_audit.md)
# ----------------------------------------------------------------------------
# Stock codes that are not merchandise (postage, fees, manual/accounting lines,
# test products). Matching is case-insensitive.
NON_PRODUCT_STOCK_CODES = frozenset({
    "POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES", "ADJUST", "ADJUST2",
    "AMAZONFEE", "CRUK", "B", "TEST001", "TEST002",
})

# The last calendar day is treated as partial when its last invoice is earlier
# than this percentile of the last-invoice time on all other days.
PARTIAL_DAY_PERCENTILE = 0.05

# ----------------------------------------------------------------------------
# Calendar knowledge available in advance (used as forecasting features)
# ----------------------------------------------------------------------------
# One-off UK public holiday inside the observed period (royal wedding).
ONE_OFF_HOLIDAYS = ("2011-04-29",)
# The business recorded no sales from 24 Dec to 3 Jan in both observed years.
YEAR_END_CLOSURE_START = (12, 24)   # (month, day)
YEAR_END_CLOSURE_END = (1, 3)

MAX_LOOKBACK = max(max(LAG_PERIODS), max(ROLLING_WINDOWS))

# ----------------------------------------------------------------------------
# Validation scheme (expanding window on the pool = everything before the test)
# ----------------------------------------------------------------------------
TEST_DAYS = FORECAST_HORIZON
N_CV_FOLDS = 4
CV_WINDOW_DAYS = 30
INTERVAL_COVERAGE = 0.80

COUNTRY_SUMMARY_FILE = PROCESSED_DIR / "country_summary.csv"
SUMMARY_FILE = PROCESSED_DIR / "summary.json"

# ----------------------------------------------------------------------------
# FROZEN chronological split (decided from the observed data BEFORE modeling):
#   * the series ends on the last FULL day, 2011-12-08 (2011-12-09 is partial);
#   * the final test set is the last 30 days;
#   * validation = 4 expanding-window folds of 30 days counted back from the
#     end of the pool (everything before the test set).
# src.train verifies these dates and refuses to run if the data disagrees.
# ----------------------------------------------------------------------------
FROZEN_SPLIT = {
    "series_start": "2009-12-01",
    "series_end": "2011-12-08",
    "pool_end": "2011-11-08",
    "test_start": "2011-11-09",
    "test_end": "2011-12-08",
    "validation_folds": [
        {"fold": 1, "train_end": "2011-07-11", "val_start": "2011-07-12", "val_end": "2011-08-10"},
        {"fold": 2, "train_end": "2011-08-10", "val_start": "2011-08-11", "val_end": "2011-09-09"},
        {"fold": 3, "train_end": "2011-09-09", "val_start": "2011-09-10", "val_end": "2011-10-09"},
        {"fold": 4, "train_end": "2011-10-09", "val_start": "2011-10-10", "val_end": "2011-11-08"},
    ],
}

HYPERPARAMETER_SEARCH_FILE = REPORTS_DIR / "hyperparameter_search.csv"
VALIDATION_FOLD_METRICS_FILE = REPORTS_DIR / "validation_fold_metrics.csv"
