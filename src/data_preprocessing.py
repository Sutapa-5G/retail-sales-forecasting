"""Loading, cleaning and daily aggregation of the Online Retail II transactions.

Design: every raw row receives exactly ONE ``status`` by walking down a fixed
"waterfall" of rules (see STATUS_RULES). Nothing is silently deleted: excluded
rows keep their status and are summarized in the cleaning log and in separate
tracking columns of the daily table (cancellations, adjustments, non-product
charges, same-day reversals).
"""
import json
from pathlib import Path

import pandas as pd

from src import config

# Waterfall order matters: a row gets the FIRST status that applies.
STATUS_RULES = {
    "sheet_overlap_duplicate": "Row appears in both workbook sheets (overlapping dates); one verified-identical copy is kept.",
    "cancellation": "Invoice number starts with 'C' (cancellation/return). Tracked separately, never netted into revenue.",
    "nonpositive_quantity": "Not a 'C' invoice but quantity <= 0 (stock write-offs, damages, adjustments). Tracked separately.",
    "nonpositive_price": "Unit price <= 0 (free items, adjustments, 'bad debt' entries).",
    "non_product": "Non-merchandise stock code (postage, fees, manual/accounting lines, test products).",
    "same_day_reversal": "Sale fully cancelled the same day by the same customer (same stock code and quantity). Treated as an order correction, not demand.",
    "partial_last_day": "Row belongs to a last calendar day that is only partly recorded.",
    "included": "Used for the daily gross revenue target.",
}
STATUS_ORDER = list(STATUS_RULES)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def read_raw_sheets(path: Path = config.RAW_FILE) -> dict:
    """Read every sheet of the raw Excel file exactly as stored."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}. See data/README.md for download instructions."
        )
    return pd.read_excel(path, sheet_name=None, engine="openpyxl")


def standardize_and_combine(sheets: dict) -> pd.DataFrame:
    """Rename columns to standard names, tag each row with its sheet, stack sheets.

    invoice_no and stock_code are converted to text because they mix numbers and
    letters. This is a type normalization, not a cleaning step: no row is dropped.
    """
    frames = []
    for sheet_name, frame in sheets.items():
        renamed = frame.rename(columns=config.COLUMN_ALIASES)
        missing = [c for c in config.STANDARD_COLUMNS if c not in renamed.columns]
        if missing:
            raise ValueError(
                f"Sheet '{sheet_name}' is missing expected columns {missing}. "
                f"Columns found: {list(frame.columns)}"
            )
        renamed = renamed[config.STANDARD_COLUMNS].copy()
        renamed["invoice_no"] = renamed["invoice_no"].astype("string")
        renamed["stock_code"] = renamed["stock_code"].astype("string")
        renamed["source_sheet"] = sheet_name
        frames.append(renamed)
    return pd.concat(frames, ignore_index=True)


def load_raw_transactions(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the raw workbook and return one combined, standardized DataFrame."""
    return standardize_and_combine(read_raw_sheets(path))


# ----------------------------------------------------------------------------
# Row-level flags
# ----------------------------------------------------------------------------
def is_cancellation(df: pd.DataFrame) -> pd.Series:
    """True where the invoice number starts with 'C' (case-insensitive)."""
    flag = df["invoice_no"].astype("string").str.upper().str.startswith("C")
    return flag.fillna(False).astype(bool)


def is_non_product(df: pd.DataFrame) -> pd.Series:
    """True where the stock code is a known non-merchandise code."""
    codes = df["stock_code"].astype("string").str.strip().str.upper()
    return codes.isin(list(config.NON_PRODUCT_STOCK_CODES)).fillna(False).astype(bool)


def _row_multiset(frame: pd.DataFrame) -> pd.Series:
    """Count of each distinct row (as text) so two row-sets can be compared."""
    return frame.astype(str).value_counts().sort_index()


def flag_sheet_overlap_duplicates(df: pd.DataFrame) -> pd.Series:
    """Flag rows of the later sheet that duplicate the earlier sheet's overlap window.

    The two workbook sheets share a date window. Before dropping anything the
    function VERIFIES that the rows of both sheets inside the window are
    identical (as multisets). If they differ it raises instead of guessing.
    """
    key_cols = config.STANDARD_COLUMNS
    drop = pd.Series(False, index=df.index)
    spans = df.groupby("source_sheet")["invoice_date"].agg(["min", "max"]).sort_values("min")
    names = list(spans.index)
    for earlier, later in zip(names[:-1], names[1:]):
        window_start = spans.loc[later, "min"]
        window_end = spans.loc[earlier, "max"]
        if window_start > window_end:
            continue  # sheets do not overlap
        in_earlier = (df["source_sheet"] == earlier) & (df["invoice_date"] >= window_start)
        in_later = (df["source_sheet"] == later) & (df["invoice_date"] <= window_end)
        if not _row_multiset(df.loc[in_earlier, key_cols]).equals(
            _row_multiset(df.loc[in_later, key_cols])
        ):
            raise ValueError(
                f"Overlap window between '{earlier}' and '{later}' contains different rows; "
                "refusing to drop either copy automatically."
            )
        drop |= in_later
    return drop


def find_same_day_reversals(
    df: pd.DataFrame, sale_mask: pd.Series, cancel_mask: pd.Series
) -> pd.Series:
    """Flag sale lines that the same customer cancelled later the same day.

    Match rule: same customer_id (known), same stock_code, cancelled quantity
    equal to the sold quantity, same calendar day, cancellation timestamp not
    before the sale. Each cancellation is used at most once (1:1 matching).
    Only information from the SAME day is used.
    """
    cols = ["customer_id", "stock_code", "quantity", "invoice_date"]
    sales = df.loc[sale_mask & df["customer_id"].notna(), cols].copy()
    sales["sale_idx"] = sales.index
    sales["day"] = sales["invoice_date"].dt.normalize()

    canc = df.loc[cancel_mask & (df["quantity"] < 0) & df["customer_id"].notna(), cols].copy()
    canc["cancel_idx"] = canc.index
    canc["day"] = canc["invoice_date"].dt.normalize()
    canc["quantity"] = -canc["quantity"]
    canc = canc.rename(columns={"invoice_date": "cancel_date"})

    pairs = sales.merge(canc, on=["customer_id", "stock_code", "day", "quantity"], how="inner")
    pairs = pairs[pairs["cancel_date"] >= pairs["invoice_date"]]
    pairs = pairs.sort_values(["invoice_date", "sale_idx", "cancel_date", "cancel_idx"])

    used_sales, used_cancels, matched = set(), set(), []
    for sale_idx, cancel_idx in zip(pairs["sale_idx"], pairs["cancel_idx"]):
        if sale_idx in used_sales or cancel_idx in used_cancels:
            continue
        used_sales.add(sale_idx)
        used_cancels.add(cancel_idx)
        matched.append(sale_idx)

    flag = pd.Series(False, index=df.index)
    flag.loc[matched] = True
    return flag


def detect_partial_last_day(df: pd.DataFrame):
    """Decide whether the last calendar day is only partly recorded.

    Compares the time of the day's last invoice with the distribution of
    last-invoice times on all earlier days. Returns (partial_day or None, evidence).
    """
    day = df["invoice_date"].dt.normalize()
    last_day = day.max()
    last_times = df.groupby(day)["invoice_date"].max()
    hours = last_times.dt.hour + last_times.dt.minute / 60.0
    earlier = hours.drop(last_day)
    threshold = float(earlier.quantile(config.PARTIAL_DAY_PERCENTILE))
    last_hour = float(hours.loc[last_day])
    partial = last_hour < threshold
    evidence = {
        "last_day": str(last_day.date()),
        "last_invoice_hour": round(last_hour, 3),
        "threshold_hour_at_percentile": round(threshold, 3),
        "percentile": config.PARTIAL_DAY_PERCENTILE,
        "earlier_days_with_last_invoice_at_or_before_this_hour": int((earlier <= last_hour).sum()),
        "earlier_days_compared": int(len(earlier)),
        "is_partial": bool(partial),
    }
    return (last_day if partial else None), evidence


# ----------------------------------------------------------------------------
# Classification (the cleaning waterfall)
# ----------------------------------------------------------------------------
def classify_transactions(df: pd.DataFrame):
    """Assign every row exactly one status. Returns (classified_df, info_dict)."""
    out = df.copy()
    out["line_value"] = out["quantity"] * out["unit_price"]
    out["status"] = "included"

    def assign(mask: pd.Series, name: str) -> None:
        out.loc[mask & (out["status"] == "included"), "status"] = name

    assign(flag_sheet_overlap_duplicates(out), "sheet_overlap_duplicate")
    assign(is_cancellation(out), "cancellation")
    assign(out["quantity"] <= 0, "nonpositive_quantity")
    assign(out["unit_price"] <= 0, "nonpositive_price")
    assign(is_non_product(out), "non_product")

    reversed_sales = find_same_day_reversals(
        out, sale_mask=out["status"] == "included", cancel_mask=out["status"] == "cancellation"
    )
    assign(reversed_sales, "same_day_reversal")

    partial_day, evidence = detect_partial_last_day(out[out["status"] != "sheet_overlap_duplicate"])
    if partial_day is not None:
        assign(out["invoice_date"].dt.normalize() == partial_day, "partial_last_day")

    out["status"] = pd.Categorical(out["status"], categories=STATUS_ORDER)
    info = {"partial_day": partial_day, "partial_day_evidence": evidence}
    return out, info


def cleaning_log(classified: pd.DataFrame) -> pd.DataFrame:
    """Waterfall table: rows and monetary value (quantity x price) per status."""
    grouped = classified.groupby("status", observed=False)["line_value"].agg(["size", "sum"])
    log = pd.DataFrame({
        "status": STATUS_ORDER,
        "rule": [STATUS_RULES[s] for s in STATUS_ORDER],
        "rows": [int(grouped.loc[s, "size"]) for s in STATUS_ORDER],
        "line_value_gbp": [round(float(grouped.loc[s, "sum"]), 2) for s in STATUS_ORDER],
    })
    log["pct_of_raw_rows"] = (log["rows"] / len(classified) * 100).round(3)
    return log


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def _per_day(frame: pd.DataFrame, column: str, how: str) -> pd.Series:
    day = frame["invoice_date"].dt.normalize()
    return getattr(frame.groupby(day)[column], how)()


def aggregate_daily(classified: pd.DataFrame, partial_day=None) -> pd.DataFrame:
    """One row per calendar day: target revenue + business and tracking metrics.

    The calendar is continuous from the first to the last FULL day. Days with no
    included sales get revenue 0 and ``is_trading_day`` = 0 (nothing is
    interpolated).
    """
    kept = classified[classified["status"] != "sheet_overlap_duplicate"]
    kept_days = kept["invoice_date"].dt.normalize()
    first_day, last_day = kept_days.min(), kept_days.max()
    if partial_day is not None:
        last_day = partial_day - pd.Timedelta(days=1)
    calendar = pd.date_range(first_day, last_day, freq="D", name=config.DATE_COLUMN)

    included = classified[classified["status"] == "included"]
    cancels = classified[classified["status"] == "cancellation"]
    adjustments = classified[classified["status"] == "nonpositive_quantity"]
    zero_price = classified[classified["status"] == "nonpositive_price"]
    non_product = classified[classified["status"] == "non_product"]
    reversals = classified[classified["status"] == "same_day_reversal"]

    daily = pd.DataFrame(index=calendar)
    daily[config.TARGET_COLUMN] = _per_day(included, "line_value", "sum")
    daily["n_lines"] = _per_day(included, "line_value", "size")
    daily["n_invoices"] = _per_day(included, "invoice_no", "nunique")
    daily["n_customers"] = _per_day(included, "customer_id", "nunique")
    daily["quantity_sold"] = _per_day(included, "quantity", "sum")
    daily["cancel_lines"] = _per_day(cancels, "line_value", "size")
    daily["cancel_invoices"] = _per_day(cancels, "invoice_no", "nunique")
    daily["cancel_value"] = _per_day(cancels.assign(line_value=cancels["line_value"].abs()), "line_value", "sum")
    daily["adjustment_lines"] = _per_day(adjustments, "line_value", "size")
    daily["zero_price_lines"] = _per_day(zero_price, "line_value", "size")
    daily["non_product_value"] = _per_day(non_product, "line_value", "sum")
    daily["same_day_reversal_value"] = _per_day(reversals, "line_value", "sum")
    daily = daily.fillna(0.0)

    count_cols = ["n_lines", "n_invoices", "n_customers", "quantity_sold", "cancel_lines",
                  "cancel_invoices", "adjustment_lines", "zero_price_lines"]
    daily[count_cols] = daily[count_cols].astype("int64")
    daily["avg_order_value"] = daily[config.TARGET_COLUMN] / daily["n_invoices"].where(daily["n_invoices"] > 0)
    daily["is_trading_day"] = (daily["n_lines"] > 0).astype("int64")
    daily[config.TARGET_COLUMN] = daily[config.TARGET_COLUMN].round(2)
    return daily.reset_index()


def aggregate_country_daily(classified: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue per country (only days with sales are stored)."""
    included = classified[classified["status"] == "included"].copy()
    included[config.DATE_COLUMN] = included["invoice_date"].dt.normalize()
    grouped = included.groupby(["country", config.DATE_COLUMN])
    out = pd.DataFrame({
        config.TARGET_COLUMN: grouped["line_value"].sum().round(2),
        "n_invoices": grouped["invoice_no"].nunique(),
        "n_lines": grouped.size(),
    }).reset_index()
    return out.astype({"country": "object"})


def country_summary(classified: pd.DataFrame) -> pd.DataFrame:
    """Totals per country over the whole included period."""
    included = classified[classified["status"] == "included"]
    grouped = included.groupby("country")
    out = pd.DataFrame({
        config.TARGET_COLUMN: grouped["line_value"].sum().round(2),
        "n_invoices": grouped["invoice_no"].nunique(),
        "n_customers": grouped["customer_id"].nunique(),
        "quantity_sold": grouped["quantity"].sum(),
    }).sort_values(config.TARGET_COLUMN, ascending=False)
    out["revenue_share_pct"] = (out[config.TARGET_COLUMN] / out[config.TARGET_COLUMN].sum() * 100).round(3)
    return out.reset_index().astype({"country": "object"})


def overall_summary(classified: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard (computed from included lines)."""
    included = classified[classified["status"] == "included"]
    return {
        "first_date": str(daily[config.DATE_COLUMN].min().date()),
        "last_date": str(daily[config.DATE_COLUMN].max().date()),
        "n_calendar_days": int(len(daily)),
        "n_trading_days": int(daily["is_trading_day"].sum()),
        "total_revenue_gbp": round(float(daily[config.TARGET_COLUMN].sum()), 2),
        "total_invoices": int(included["invoice_no"].nunique()),
        "total_lines": int(len(included)),
        "unique_customers": int(included["customer_id"].nunique()),
        "share_lines_missing_customer_id": round(float(included["customer_id"].isna().mean()), 5),
        "total_quantity_sold": int(included["quantity"].sum()),
    }


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_pipeline(raw_path: Path = config.RAW_FILE, save: bool = True) -> dict:
    """Raw workbook -> classified rows -> daily tables. Optionally writes data/processed/."""
    raw = load_raw_transactions(raw_path)
    classified, info = classify_transactions(raw)
    daily = aggregate_daily(classified, info["partial_day"])
    results = {
        "raw_rows": int(len(raw)),
        "classified": classified,
        "info": info,
        "log": cleaning_log(classified),
        "daily": daily,
        "country_daily": aggregate_country_daily(classified),
        "country_summary": country_summary(classified),
        "summary": overall_summary(classified, daily),
    }
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(config.DAILY_FILE, index=False)
        results["country_daily"].to_csv(config.COUNTRY_DAILY_FILE, index=False)
        results["country_summary"].to_csv(config.COUNTRY_SUMMARY_FILE, index=False)
        results["log"].to_csv(config.CLEANING_LOG_FILE, index=False)
        summary = dict(results["summary"])
        summary["raw_rows"] = results["raw_rows"]
        summary["partial_day_evidence"] = info["partial_day_evidence"]
        config.SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    return results


# ----------------------------------------------------------------------------
# Readers for processed files
# ----------------------------------------------------------------------------
def load_daily_metrics(path: Path = config.DAILY_FILE) -> pd.DataFrame:
    """Read the processed daily table (dates parsed)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed daily file not found: {path}. Run the pipeline first.")
    return pd.read_csv(path, parse_dates=[config.DATE_COLUMN])


def to_revenue_series(daily: pd.DataFrame) -> pd.Series:
    """Daily revenue as a Series with a continuous daily DatetimeIndex."""
    series = daily.set_index(config.DATE_COLUMN)[config.TARGET_COLUMN].astype(float)
    series.index = pd.DatetimeIndex(series.index, freq="D")
    series.name = config.TARGET_COLUMN
    return series


def load_revenue_series(path: Path = config.DAILY_FILE) -> pd.Series:
    """Convenience: processed daily file -> revenue Series."""
    return to_revenue_series(load_daily_metrics(path))


def main() -> None:
    """``python -m src.data_preprocessing``: build all processed files from the raw workbook."""
    results = run_pipeline()
    print(results["log"][["status", "rows", "line_value_gbp"]].to_string(index=False))
    print("\nDaily table:", results["daily"].shape,
          "| period:", results["summary"]["first_date"], "->", results["summary"]["last_date"])
    print("Written to:", config.PROCESSED_DIR.relative_to(config.PROJECT_ROOT))


if __name__ == "__main__":
    main()
