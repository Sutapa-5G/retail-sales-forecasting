# Data

## Dataset

**Online Retail II**: real transactions of a UK-based online retailer (non-store, mainly gift-ware, many wholesale customers), covering 2009-12-01 to 2011-12-09.

| | |
|---|---|
| Source | UCI Machine Learning Repository |
| Page | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| Direct download | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip |
| DOI | https://doi.org/10.24432/C5CG6D |
| License | **Creative Commons Attribution 4.0 International (CC BY 4.0)** |
| Citation | Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D |

The project **code** is MIT licensed (see `LICENSE`). The **dataset** keeps its own CC BY 4.0 license; use of the data requires the attribution above.

## Raw data is intentionally NOT included in this repository

The raw workbook is about 45 MB and can be downloaded from the official source, so it is excluded (`.gitignore`). Everything in the repository works once you place the file in `data/raw/`.

### How to get it

1. Download the zip from the direct link above (or from the dataset page, button "Download").
2. Unzip it. It contains one file: `online_retail_II.xlsx`.
3. Put that file at `data/raw/online_retail_II.xlsx`.

Command line (macOS/Linux):

```bash
curl -L -o online_retail_ii.zip "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
unzip online_retail_ii.zip -d data/raw
```

Windows PowerShell:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip" -OutFile online_retail_ii.zip
Expand-Archive online_retail_ii.zip -DestinationPath data/raw
```

### Verify the file

The results in this repository were produced from a file with this SHA-256 (UCI does not publish a checksum, so this is a record of the exact file used):

```
bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980
```

```bash
sha256sum data/raw/online_retail_II.xlsx          # macOS: shasum -a 256 ...
```

Expected structure: two sheets (`Year 2009-2010` with 525,461 rows and `Year 2010-2011` with 541,910 rows) and the headers `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`.

## Processed files (included, small)

`python -m src.data_preprocessing` rebuilds these from the raw file. They are aggregates derived from the dataset (shared under the same CC BY 4.0 attribution):

| File | Contents |
|---|---|
| `processed/daily_metrics.csv` | one row per day: revenue (the target), invoices, customers, quantity, cancellations, and tracked excluded values |
| `processed/country_daily.csv` | daily revenue per country |
| `processed/country_summary.csv` | totals and revenue share per country |
| `processed/cleaning_log.csv` | the cleaning waterfall (rows and value per rule) |
| `processed/summary.json` | headline numbers used by the dashboard |
| `processed/test_predictions.csv`, `validation_predictions.csv`, `forecast_30d.csv` | model outputs written by `python -m src.train` |
