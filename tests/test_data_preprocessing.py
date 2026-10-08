"""Data loading, revenue calculation, cleaning rules and daily aggregation."""
import pandas as pd
import pytest
from openpyxl import Workbook

from conftest import make_transactions
from src import config
from src import data_preprocessing as dp


# ---------------------------------------------------------------- loading
def _write_workbook(path, sheets):
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
        for row in rows:
            ws.append(row)
    wb.save(path)


def test_load_raw_transactions_reads_real_header_names(tmp_path):
    path = tmp_path / "retail.xlsx"
    ts = pd.Timestamp("2010-01-04 09:00").to_pydatetime()
    _write_workbook(path, {"Year A": [[489434, 85048, "X", 12, ts, 6.95, 13085, "United Kingdom"],
                                      ["C489449", "79323P", "Y", -1, ts, 2.5, None, "France"]]})
    df = dp.load_raw_transactions(path)
    assert list(df.columns) == config.STANDARD_COLUMNS + ["source_sheet"]
    assert len(df) == 2
    assert df["invoice_no"].tolist() == ["489434", "C489449"]       # numbers and text both become text
    assert df["source_sheet"].tolist() == ["Year A", "Year A"]


def test_load_raw_transactions_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dp.load_raw_transactions(tmp_path / "nope.xlsx")


def test_standardize_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing expected columns"):
        dp.standardize_and_combine({"s": pd.DataFrame({"Invoice": [1]})})


# ---------------------------------------------------------------- flags and revenue
def test_revenue_is_quantity_times_price():
    df = make_transactions([("1", "A1", 3, 2.5, "2010-01-04 09:00", 1, "UK")])
    classified, _ = dp.classify_transactions(df)
    assert classified.loc[0, "line_value"] == pytest.approx(7.5)


def test_cancellation_flag_is_case_insensitive():
    df = make_transactions([("C100", "A", -1, 1, "2010-01-04 09:00", 1, "UK"),
                            ("c101", "A", -1, 1, "2010-01-04 09:00", 1, "UK"),
                            ("102", "A", 1, 1, "2010-01-04 09:00", 1, "UK")])
    assert dp.is_cancellation(df).tolist() == [True, True, False]


# ---------------------------------------------------------------- cleaning waterfall
def _waterfall_frame():
    return make_transactions([
        ("1", "A1", 2, 10.0, "2010-01-04 09:00", 1, "UK"),        # included
        ("C2", "A1", -1, 10.0, "2010-01-04 10:00", 1, "UK"),      # cancellation
        ("3", "A1", -5, 10.0, "2010-01-04 11:00", 1, "UK"),       # non-C negative quantity
        ("4", "A1", 1, 0.0, "2010-01-04 12:00", 1, "UK"),         # zero price
        ("5", "A1", 1, -9.0, "2010-01-04 12:30", 1, "UK"),        # negative price
        ("6", "POST", 1, 15.0, "2010-01-04 13:00", 1, "UK"),      # non-product
        ("7", "post", 1, 15.0, "2010-01-04 13:05", 1, "UK"),      # non-product, lower case
        ("8", "P9", 0, 5.0, "2010-01-04 14:00", 1, "UK"),         # zero quantity
        ("9", "DCGS0058", 1, 4.0, "2010-01-04 15:00", 1, "UK"),   # shop product: must stay included
        ("10", "A1", 1, 1.0, "2010-01-04 18:00", 1, "UK"),        # included, day continues late
        ("11", "A1", 1, 1.0, "2010-01-05 17:00", 1, "UK"),
        ("12", "A1", 1, 1.0, "2010-01-06 17:30", 1, "UK"),
        ("13", "A1", 1, 1.0, "2010-01-07 17:15", 1, "UK"),
        ("14", "A1", 1, 1.0, "2010-01-08 17:45", 1, "UK"),
    ])


def test_every_row_gets_exactly_one_status_and_no_row_is_lost():
    df = _waterfall_frame()
    classified, _ = dp.classify_transactions(df)
    assert len(classified) == len(df)
    assert classified["status"].notna().all()
    assert classified["status"].isin(dp.STATUS_ORDER).all()


def test_cleaning_rules_assign_expected_statuses():
    classified, _ = dp.classify_transactions(_waterfall_frame())
    status = classified.set_index("invoice_no")["status"].astype(str)
    assert status["1"] == "included"
    assert status["C2"] == "cancellation"
    assert status["3"] == "nonpositive_quantity"
    assert status["8"] == "nonpositive_quantity"
    assert status["4"] == "nonpositive_price"
    assert status["5"] == "nonpositive_price"
    assert status["6"] == "non_product"
    assert status["7"] == "non_product"
    assert status["9"] == "included"


def test_cleaning_log_reconciles_to_raw_rows():
    df = _waterfall_frame()
    classified, _ = dp.classify_transactions(df)
    log = dp.cleaning_log(classified)
    assert log["rows"].sum() == len(df)


# ---------------------------------------------------------------- sheet overlap
def _two_sheets(identical=True):
    a = make_transactions([("1", "A", 1, 1.0, "2010-12-01 09:00", 1, "UK"),
                           ("2", "A", 1, 1.0, "2010-12-05 09:00", 1, "UK"),
                           ("3", "A", 1, 1.0, "2010-12-09 09:00", 1, "UK")])
    b = make_transactions([("2", "A", 1, 1.0 if identical else 9.0, "2010-12-05 09:00", 1, "UK"),
                           ("3", "A", 1, 1.0, "2010-12-09 09:00", 1, "UK"),
                           ("4", "A", 1, 1.0, "2010-12-20 09:00", 1, "UK")])
    # sheet A also needs to start before sheet B for the window to be [B.min, A.max]
    a.loc[0, "invoice_date"] = pd.Timestamp("2010-11-30 09:00")
    a["source_sheet"], b["source_sheet"] = "early", "late"
    return pd.concat([a, b], ignore_index=True)


def test_sheet_overlap_flags_only_later_sheet_copy():
    df = _two_sheets(identical=True)
    flag = dp.flag_sheet_overlap_duplicates(df)
    assert flag.sum() == 2
    assert (df.loc[flag, "source_sheet"] == "late").all()
    assert df.loc[flag, "invoice_no"].tolist() == ["2", "3"]


def test_sheet_overlap_refuses_when_copies_differ():
    with pytest.raises(ValueError, match="different rows"):
        dp.flag_sheet_overlap_duplicates(_two_sheets(identical=False))


def test_genuine_repeated_lines_within_a_sheet_are_not_dropped():
    row = ("1", "A", 1, 1.0, "2010-01-04 09:00", 1, "UK")
    classified, _ = dp.classify_transactions(make_transactions([row, row]))
    assert (classified["status"].astype(str) == "included").sum() == 2


# ---------------------------------------------------------------- same-day reversal
def test_same_day_reversal_matches_one_to_one():
    df = make_transactions([
        ("10", "A", 100, 1.0, "2010-03-01 10:00", 7, "UK"),     # sale ...
        ("C11", "A", -100, 1.0, "2010-03-01 10:15", 7, "UK"),   # ... cancelled same day
        ("12", "A", 5, 1.0, "2010-03-01 10:00", 8, "UK"),       # different customer: untouched
        ("13", "P2", 50, 1.0, "2010-03-02 10:00", 7, "UK"),      # cancelled NEXT day: untouched
        ("C14", "P2", -50, 1.0, "2010-03-03 10:00", 7, "UK"),
        ("15", "P3", 20, 1.0, "2010-03-04 10:00", 9, "UK"),      # two sales, one cancellation
        ("16", "P3", 20, 1.0, "2010-03-04 10:05", 9, "UK"),
        ("C17", "P3", -20, 1.0, "2010-03-04 10:30", 9, "UK"),
        ("18", "P4", 3, 1.0, "2010-03-05 10:00", None, "UK"),    # unknown customer: never matched
        ("C19", "P4", -3, 1.0, "2010-03-05 10:10", None, "UK"),
    ])
    classified, _ = dp.classify_transactions(df)
    status = classified.set_index("invoice_no")["status"].astype(str)
    assert status["10"] == "same_day_reversal"
    assert status["12"] == "included"
    assert status["13"] == "included"
    assert status["15"] == "same_day_reversal"      # earliest sale is matched
    assert status["16"] == "included"               # the second sale stays
    assert status["18"] == "included"


def test_cancellation_before_sale_is_not_a_reversal():
    df = make_transactions([("C1", "A", -10, 1.0, "2010-03-01 09:00", 7, "UK"),
                            ("2", "A", 10, 1.0, "2010-03-01 11:00", 7, "UK")])
    classified, _ = dp.classify_transactions(df)
    assert classified.set_index("invoice_no").loc["2", "status"] == "included"


# ---------------------------------------------------------------- partial last day
def test_partial_last_day_is_detected_and_excluded():
    rows = [(str(i), "A", 1, 1.0, f"2010-01-{4 + i:02d} 17:30", 1, "UK") for i in range(1, 20)]
    rows.append(("99", "A", 1, 1.0, "2010-01-30 11:00", 1, "UK"))    # last day stops at 11:00
    df = make_transactions(rows)
    classified, info = dp.classify_transactions(df)
    assert info["partial_day"] == pd.Timestamp("2010-01-30")
    daily = dp.aggregate_daily(classified, info["partial_day"])
    assert daily["date"].max() == pd.Timestamp("2010-01-29")


def test_normal_last_day_is_kept():
    rows = [(str(i), "A", 1, 1.0, f"2010-01-{4 + i:02d} 17:30", 1, "UK") for i in range(1, 21)]
    classified, info = dp.classify_transactions(make_transactions(rows))
    assert info["partial_day"] is None


# ---------------------------------------------------------------- daily aggregation
def test_daily_aggregation_values_and_zero_filled_days():
    df = make_transactions([
        ("1", "A", 2, 10.0, "2010-01-04 09:00", 1, "UK"),       # 20
        ("1", "P2", 1, 5.0, "2010-01-04 09:00", 1, "UK"),        # 5 (same invoice)
        ("2", "A", 4, 2.5, "2010-01-04 10:00", 2, "France"),    # 10
        ("C3", "A", -1, 10.0, "2010-01-04 11:00", 1, "UK"),     # partial cancellation: tracked, NOT netted
        ("4", "A", 1, 100.0, "2010-01-06 17:00", 3, "UK"),      # 100 on Wednesday; Tuesday has no rows
        ("5", "A", 1, 1.0, "2010-01-07 17:00", 3, "UK"),
        ("6", "A", 1, 1.0, "2010-01-08 17:00", 3, "UK"),
    ])
    classified, info = dp.classify_transactions(df)
    daily = dp.aggregate_daily(classified, info["partial_day"]).set_index("date")
    mon = daily.loc["2010-01-04"]
    assert mon["revenue"] == pytest.approx(35.0)
    assert mon["n_invoices"] == 2 and mon["n_customers"] == 2 and mon["n_lines"] == 3
    assert mon["quantity_sold"] == 7
    assert mon["cancel_lines"] == 1 and mon["cancel_value"] == pytest.approx(10.0)
    assert mon["avg_order_value"] == pytest.approx(17.5)
    tue = daily.loc["2010-01-05"]
    assert tue["revenue"] == 0 and tue["is_trading_day"] == 0
    assert daily.loc["2010-01-06", "revenue"] == pytest.approx(100.0)
    # continuous calendar
    assert (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_country_aggregates_sum_to_daily_revenue():
    df = make_transactions([("1", "A", 2, 10.0, "2010-01-04 09:00", 1, "UK"),
                            ("2", "A", 1, 30.0, "2010-01-04 10:00", 2, "France"),
                            ("3", "A", 1, 1.0, "2010-01-05 17:00", 2, "France"),
                            ("4", "A", 1, 1.0, "2010-01-06 17:00", 2, "France"),
                            ("5", "A", 1, 1.0, "2010-01-07 17:00", 2, "France")])
    classified, info = dp.classify_transactions(df)
    daily = dp.aggregate_daily(classified, info["partial_day"])
    by_country = dp.aggregate_country_daily(classified)
    assert by_country["revenue"].sum() == pytest.approx(daily["revenue"].sum())
    summary = dp.country_summary(classified)
    assert summary["revenue_share_pct"].sum() == pytest.approx(100.0, abs=0.01)
