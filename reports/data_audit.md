# Data Audit Report

All numbers below were produced by running the code in this repository on the official UCI file (`online_retail_II.xlsx`). Reproduce them with `notebooks/01_data_exploration.ipynb` and `python -m src.data_preprocessing`.

## 1. Source and raw dimensions

* Dataset: UCI *Online Retail II* (CC BY 4.0), https://archive.ics.uci.edu/dataset/502/online+retail+ii
* Workbook with **2 sheets**: `Year 2009-2010` (525,461 rows), `Year 2010-2011` (541,910 rows)
* **Raw rows: 1,067,371**, 8 data columns (plus a `source_sheet` column added by the loader)
* The real file headers are `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` (the UCI web page documents `InvoiceNo`, `UnitPrice`, `CustomerID`; the loader accepts both and standardizes the names).

Three levels of data are kept distinct throughout the project:

| Level | What it is | Size |
|---|---|---|
| **Raw observations** | one row per invoice line as stored in the workbook | 1,067,371 rows |
| **Aggregated observations** | the cleaned lines summed per calendar day (`data/processed/daily_metrics.csv`) | 738 days |
| **Forecast target** | `revenue` = daily gross revenue in GBP (definition in section 9) | one value per day |

## 2. Columns, types and missing values

| column | dtype | missing | missing % |
|---|---|---|---|
| invoice_no | string | 0 | 0.00 |
| stock_code | string | 0 | 0.00 |
| description | object | 4,382 | 0.41 |
| quantity | int64 | 0 | 0.00 |
| invoice_date | datetime64[us] | 0 | 0.00 |
| unit_price | float64 | 0 | 0.00 |
| customer_id | float64 | 243,007 | 22.77 |
| country | str | 0 | 0.00 |

Missing customer IDs do not affect revenue and are kept; the *unique customers* metric counts only rows with an ID (22.3% of included lines have none). Most missing descriptions belong to zero-price adjustment rows.

## 3. Date range, granularity and gaps

* Granularity: one row per invoice line with a timestamp (minute resolution); modeled at **daily** level.
* Sheet date ranges: `Year 2009-2010`: 2009-12-01 07:45:00 to 2010-12-09 20:01:00; `Year 2010-2011`: 2010-12-01 08:26:00 to 2011-12-09 12:50:00.
* Combined range after removing the sheet overlap: **2009-12-01 07:45:00** to **2011-12-09 12:50:00**.
* Calendar days in range: **739**; days with at least one row: **604**; days with no rows: **135** (104 Saturdays and 31 other days).
* The 31 non-Saturday no-row days are UK public holidays (Easter weekend, the early-May, spring and summer bank holidays, the one-off holiday on 2011-04-29) and the 24 December - 3 January window in both years.
* Only 1 Saturday has any trading in the whole period.
* **Last day is partial.** The final day, 2011-12-09, ends at hour 12.83 (12:50), earlier than the 5th percentile of last-invoice times on other days (16.02h); 0 of 603 earlier days ended that early. It is excluded from the modeled series.

## 4. Sheet overlap and duplicates

* The two sheets overlap between 2010-12-01 08:26:00 and 2010-12-09 20:01:00. The loader **verifies** that both sheets hold identical rows in that window before dropping the later sheet's copy: **22,523 rows** (gross value £438,852.65 counted twice if kept).
* Extra copies of fully identical rows in the raw data: **34,335**, of which 12,133 repeat inside the same sheet; the rest come from the sheet overlap.
* After removing the overlap, **11,812** exact duplicate extra rows remain inside sheets. They are **kept**: a repeated scan of the same product on one invoice is legitimate and cannot be told apart from a data-entry duplicate. Among included lines, 11,689 are exact duplicates worth £56,473.56 (0.29% of included revenue), so the decision has a small effect.

## 5. Cancellations and returns

* Rows on `C` invoices: **19,494** (1.83% of raw rows); 19,493 have negative quantity and 1 has a positive quantity.
* Treatment: tracked separately (`cancel_lines`, `cancel_invoices`, `cancel_value`) and **never netted into the revenue target**. A return is booked on the day it is processed, often long after the sale, so netting would put negative revenue on days unrelated to demand.

## 6. Negative and zero quantities and prices

* Rows that are not `C` invoices but have quantity <= 0: **3,457** (stock write-offs and adjustments; 2,689 have no description). Excluded, tracked in `adjustment_lines`.
* Rows with unit price = 0: 6,202; with unit price < 0: 5 ("Adjust bad debt" accounting entries).
* Quantity: median 3, 1st percentile -3, 99th percentile 100, min -80,995, max 80,995.
* Unit price: median 2.10, 99th percentile 18.00, max 38,970.00.

## 7. Suspicious and non-product rows

**Non-product stock codes** (postage, fees, manual/accounting lines, test products) are excluded from the target and tracked in `non_product_value`:

| stock code | rows | example_description | value_gbp |
|---|---|---|---|
| POST | 1,851 | POSTAGE | 125,682.42 |
| DOT | 1,415 | DOTCOM POSTAGE | 309,854.11 |
| M | 856 | Manual | 339,614.89 |
| C2 | 267 | CARRIAGE | 13,426.00 |
| ADJUST | 36 | Adjustment by john on 26/01/2010 16 | 8,897.93 |
| BANK CHARGES | 34 |  Bank Charges | 519.24 |
| TEST001 | 9 | This is a test product. | 225.00 |
| D | 5 | Discount | 397.89 |
| m | 5 | Manual | 15.05 |
| ADJUST2 | 3 | Adjustment by Peter on Jun 25 2010  | 731.05 |
| S | 3 | SAMPLES | 136.85 |
| AMAZONFEE | 3 | AMAZON FEE | 20,467.80 |
| B | 1 | Adjust bad debt | 11,062.06 |
| TEST002 | 1 | This is a test product. | 1.00 |

Non-standard codes that are real products or sales (for example `DCGS…` shop items, `gift_0001_…` vouchers, `PADS`) are kept.

**Orders reversed the same day.** 1,250 sale lines worth £340,429.66 were cancelled by the same customer, for the same stock code and quantity, later the same day. They are excluded and tracked in `same_day_reversal_value`. The largest raw "orders" in the file are of this kind, for example a 74,215-unit line of £77,183.60 on 2011-01-18 that was cancelled 16 minutes later. Left in, it would make that day more than five times busier than it was and distort 28 days of lag features.

## 8. Cleaning waterfall (every raw row gets exactly one status)

| status | rows | line_value_gbp | pct_of_raw_rows |
|---|---|---|---|
| sheet_overlap_duplicate | 22,523 | 377,488.45 | 2.11 |
| cancellation | 19,165 | -1,465,303.66 | 1.796 |
| nonpositive_quantity | 3,393 | 0.00 | 0.318 |
| nonpositive_price | 2,636 | -158,676.14 | 0.247 |
| non_product | 4,489 | 831,031.29 | 0.421 |
| same_day_reversal | 1,250 | 340,429.66 | 0.117 |
| partial_last_day | 1,617 | 29,644.66 | 0.151 |
| included | 1,012,298 | 19,332,636.31 | 94.84 |

Rows sum to the raw total: **1,067,371 = 1,067,371**.

Rules (first matching status wins):

| status | rule |
|---|---|
| sheet_overlap_duplicate | Row appears in both workbook sheets (overlapping dates); one verified-identical copy is kept. |
| cancellation | Invoice number starts with 'C' (cancellation/return). Tracked separately, never netted into revenue. |
| nonpositive_quantity | Not a 'C' invoice but quantity <= 0 (stock write-offs, damages, adjustments). Tracked separately. |
| nonpositive_price | Unit price <= 0 (free items, adjustments, 'bad debt' entries). |
| non_product | Non-merchandise stock code (postage, fees, manual/accounting lines, test products). |
| same_day_reversal | Sale fully cancelled the same day by the same customer (same stock code and quantity). Treated as an order correction, not demand. |
| partial_last_day | Row belongs to a last calendar day that is only partly recorded. |
| included | Used for the daily gross revenue target. |

## 9. Final modeling dataset and the target

* **Included lines (used for the target): 1,012,298** of 1,067,371 raw rows.
* **Target definition:** `revenue` = sum of `Quantity x Price` over included lines for each calendar date (date part of `InvoiceDate`), all countries combined, in GBP.
* Total gross revenue in the daily table: **£19,332,636.29**.
* Tracked but not in the target: cancellations (absolute value £1,297,261.73), non-product charges (£828,206.95), same-day reversals (£171,960.06).

## 10. Final daily dataset

* File: `data/processed/daily_metrics.csv`, **738 rows** (2009-12-01 to 2011-12-08), **603 trading days** and 135 non-trading days (revenue = 0, nothing interpolated).
* Columns: `date`, `revenue`, `n_lines`, `n_invoices`, `n_customers`, `quantity_sold`, `cancel_lines`, `cancel_invoices`, `cancel_value`, `adjustment_lines`, `zero_price_lines`, `non_product_value`, `same_day_reversal_value`, `avg_order_value`, `is_trading_day`.
* Also written: `country_daily.csv`, `country_summary.csv` (43 countries), `cleaning_log.csv`, `summary.json`.
* Headline numbers: 39,358 invoices, 5,850 unique customers (known IDs), total quantity 10,998,462.

## 11. Frozen chronological split (exact dates)

Decided from the calendar before any modeling: the series ends on the last *full* day, and the final test set is the last 30 days.

| Part | Dates |
|---|---|
| Full series | 2009-12-01 to 2011-12-08 (738 days) |
| **Pool** (all exploration, validation and selection) | 2009-12-01 to 2011-11-08 |
| **Final test set** (untouched until the single final evaluation) | 2011-11-09 to 2011-12-08 (30 days) |
| Validation fold 1 | train through 2011-07-11, validate 2011-07-12 to 2011-08-10 |
| Validation fold 2 | train through 2011-08-10, validate 2011-08-11 to 2011-09-09 |
| Validation fold 3 | train through 2011-09-09, validate 2011-09-10 to 2011-10-09 |
| Validation fold 4 | train through 2011-10-09, validate 2011-10-10 to 2011-11-08 |

## 12. Honest notes on the audit

* The audit describes **all** rows (including the test-period days); no cleaning rule was chosen by looking at test-period *revenue*. The rules respond to data-quality evidence (cancellation prefix, non-positive values, sheet overlap, partial day, same-day reversals).
* The same-day reversal rule changes £171,960 (0.9% of revenue) and is a judgement call. It is documented, narrow, and tracked daily so it can be reversed.
* Keeping within-sheet exact duplicates is also a judgement call (see section 4).
