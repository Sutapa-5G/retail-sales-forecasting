# Business Insights

Every finding below comes from the data in this repository and the numbers are reproduced by the notebooks. Findings are labelled by type:

* **Observed**: a historical pattern in the data.
* **Model**: an output of the saved forecasting model (an estimate, not a guarantee).
* **Interpretation**: a cautious business reading. Correlation in this data does not establish causes, and the data cannot say *why* a pattern occurred.

Revenue means **daily gross product revenue in GBP, all countries** (cancellations, non-product charges and same-day reversed orders excluded; see `reports/data_audit.md`). Period: 2009-12-01 to 2011-12-08.

## 1. Scale of the business
* **Observed.** Gross revenue over the 738-day period is **£19,332,636** from 39,358 invoices on 603 trading days: about **£32,061 per trading day** (median £27,974).
* **Observed.** The typical invoice is worth about £302 (median) and £491 on average; the average being higher shows a tail of large orders.

## 2. A strong autumn peak
* **Observed.** In 2010, September to November delivered **36%** of the calendar year's revenue, and November alone **15%**. Revenue per trading day was £55,031 in November 2010 versus £22,472 in February 2010.
* **Interpretation.** The business is highly seasonal, so inventory and staffing built on a yearly average will be too low in autumn and too high in winter. Only two full Decembers (2009 and 2010) exist in the data, so the December pattern the forecast relies on is learned from very little history.

## 3. Modest growth
* **Observed.** Revenue for January to November 2011 was **+2.3%** versus January to November 2010; September to November was **+5.5%**; November 2011 alone was +1.4% versus November 2010.
* **Interpretation.** Year-on-year growth is small relative to the day-to-day noise. The data does not support expecting large growth without new information.

## 4. A fixed weekly and holiday rhythm
* **Observed.** Saturday is effectively closed: **104 of 105 Saturdays** had no sales at all. Sunday trades at roughly **52%** of the average weekday level.
* **Observed.** The remaining 31 zero-sales days are UK public holidays and the 24 December - 3 January window in both observed years. That is why the model treats these as closed.
* **Interpretation.** Zero-revenue days are predictable from the calendar and should be planned for, not treated as demand collapses.

## 5. Day-to-day volatility
* **Observed.** On trading days the standard deviation is about **52%** of the mean: a typical day can be far above or below the average. The strongest single day in the period is £111,531 on 2011-11-14.
* **Interpretation.** Any daily forecast will be noisy. Weekly or monthly planning is more reliable than day-level targets.

## 6. Cancellations and returns
* **Observed.** Cancelled value (invoices starting with `C`) equals **6.7%** of gross revenue, and cancellation invoices are about 17% of all invoices (cancellation + sales invoices).
* **Observed.** The monthly cancelled-value ratio is usually between 3% and 11% of gross revenue (median 6.2%). January 2011 stands out at **22%**, but a single cancelled order of £77,183.60 accounts for most of the excess: without it the ratio would be 9%.
* **Interpretation.** Returns are a steady cost of doing business, and single very large orders that are cancelled can dominate a month's statistics. This is why returns are tracked separately rather than netted into demand.

## 7. Customers
* **Observed.** 5,850 distinct customer IDs account for 87% of revenue; the rest (22% of lines, 13% of revenue) has no customer ID.
* **Observed.** Among identified customers, the top 20% generate **77%** of revenue, while the top 10 customers generate 16%. About **72%** of customers placed more than one invoice.
* **Interpretation.** Revenue is spread across many customers (no single dominant account) but skewed towards repeat buyers. Missing IDs mean customer-level analysis covers most, not all, of the business.

## 8. Geographic concentration
* **Observed.** The United Kingdom provides **85.4%** of revenue; the top five countries together 95.0% (across 43 countries). EIRE ranks second (3.2%) with only 3 customer IDs.
* **Interpretation.** The business is overwhelmingly UK-based; a few international accounts, such as EIRE's, can make a country's numbers lumpy.

## 9. What the forecast model implies
* **Model.** The selected model (XGBoost (n=400, depth=5, lr=0.03)) reaches a **30-day recursive test MAE of £8,674** (WAPE 17.5%), versus £12,763 for the best baseline on test (Seasonal naive (same weekday last week)), i.e. **32% lower**. On validation it beat the best baseline (Seasonal average (same weekday, last 4 weeks)) by 20% on average, but not in every fold.
* **Model.** The 30-day forecast (2011-12-09 to 2012-01-07) totals **£621,338** with 17 trading days and 13 zero days (holidays, the year-end closure and closed Saturdays).
* **Interpretation (caution).** The forecast level for 9-23 December averages £42,923 per trading day versus £32,099 on the same dates in 2010 (**+34%**), far more than the observed year-on-year growth. Treat the December level as an upper-leaning estimate; the *shape* (busy until about 23 December, closed afterwards, a quieter January restart) is the more dependable message for planning.
* **Model.** The 80% interval is an empirical approximation; it contained 83% of the 30 test days.
* **Not claimed.** The forecast is not a guarantee of future sales, and nothing here shows what *causes* strong or weak days.
