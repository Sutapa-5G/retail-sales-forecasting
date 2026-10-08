"""Retail Sales Forecasting & Analytics Dashboard.

Run from the project root:  streamlit run app/app.py

The app only LOADS saved artifacts (processed CSVs, the saved model and its
metadata). It never retrains a model and never fits preprocessing.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import data_preprocessing as dp  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import forecast as fc  # noqa: E402

st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📈", layout="wide")

REQUIRED_FILES = [config.DAILY_FILE, config.COUNTRY_DAILY_FILE, config.COUNTRY_SUMMARY_FILE, config.SUMMARY_FILE,
                  config.MODEL_FILE, config.MODEL_METADATA_FILE, config.MODEL_COMPARISON_FILE,
                  config.TEST_PREDICTIONS_FILE, config.REPORTS_DIR / "insights.md"]
COLORS = {"actual": "#1f2a44", "forecast": "#d1495b", "band": "rgba(209,73,91,0.18)", "accent": "#00798c",
          "muted": "#8d99ae"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------------------
# Cached loaders (artifacts only; no training happens here)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_daily() -> pd.DataFrame:
    return dp.load_daily_metrics()


@st.cache_data(show_spinner=False)
def load_country_daily() -> pd.DataFrame:
    return pd.read_csv(config.COUNTRY_DAILY_FILE, parse_dates=[config.DATE_COLUMN])


@st.cache_data(show_spinner=False)
def load_country_summary() -> pd.DataFrame:
    return pd.read_csv(config.COUNTRY_SUMMARY_FILE)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(config.MODEL_COMPARISON_FILE)


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(config.TEST_PREDICTIONS_FILE, parse_dates=[config.DATE_COLUMN])


@st.cache_data(show_spinner=False)
def load_insights_text() -> str:
    return (config.REPORTS_DIR / "insights.md").read_text()


@st.cache_resource(show_spinner=False)
def load_model():
    return fc.load_artifacts()


@st.cache_data(show_spinner=False)
def make_forecast(horizon: int) -> pd.DataFrame:
    """Forecast from the saved model. Cached per horizon; never retrains."""
    model, meta = load_model()
    history = dp.to_revenue_series(load_daily())
    return fc.forecast_with_artifacts(model, meta, history, horizon)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def gbp(value: float) -> str:
    return f"£{value:,.0f}"


def style_figure(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      yaxis_tickprefix="£", yaxis_tickformat=",.0f")
    return fig


def aggregate(frame: pd.DataFrame, level: str) -> pd.Series:
    series = frame.set_index(config.DATE_COLUMN)[config.TARGET_COLUMN]
    if level == "Weekly":
        return series.resample("W-SUN").sum()
    if level == "Monthly":
        return series.resample("MS").sum()
    return series


# ----------------------------------------------------------------------------
# Startup checks
# ----------------------------------------------------------------------------
missing = [str(p.relative_to(config.PROJECT_ROOT)) for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error("Some project artifacts are missing, so the dashboard cannot start.")
    st.write("Missing files:", missing)
    st.code("python -m src.data_preprocessing   # builds data/processed from data/raw\n"
            "python -m src.train                # trains, evaluates and saves the model", language="bash")
    st.stop()

daily_all = load_daily()
summary = load_json(str(config.SUMMARY_FILE))
model, meta = load_model()
data_min, data_max = daily_all[config.DATE_COLUMN].min().date(), daily_all[config.DATE_COLUMN].max().date()

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.title("Controls")
date_range = st.sidebar.date_input("Date range (historical views)", value=(data_min, data_max),
                                   min_value=data_min, max_value=data_max)
if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.info("Please select both a start and an end date in the sidebar.")
    st.stop()
start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
if start_date > end_date:
    st.warning("The start date is after the end date. Please correct the date range.")
    st.stop()

country_summary_all = load_country_summary()
default_countries = country_summary_all["country"].head(5).tolist()
selected_countries = st.sidebar.multiselect("Countries (country analysis)", country_summary_all["country"].tolist(),
                                            default=default_countries)
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=config.FORECAST_HORIZON,
                            value=config.FORECAST_HORIZON,
                            help="The model was validated for up to 30 days ahead.")
st.sidebar.caption(f"Data: {data_min} to {data_max}. The forecast always starts the day after the last full day, "
                   "whatever date range is selected.")

view = daily_all[(daily_all[config.DATE_COLUMN] >= start_date) & (daily_all[config.DATE_COLUMN] <= end_date)].copy()
if view.empty:
    st.warning("No data in the selected date range.")
    st.stop()

st.title("📈 Retail Sales Forecasting & Analytics Dashboard")
st.caption("UCI Online Retail II · daily gross revenue in GBP, all countries · "
           "forecasts are model estimates, not guarantees")

tabs = st.tabs(["Executive Overview", "Historical Sales", "Sales Analysis", "Forecast",
                "Model Performance", "Business Insights"])

# ----------------------------------------------------------------------------
# 1. Executive overview
# ----------------------------------------------------------------------------
with tabs[0]:
    trading_view = view[view["is_trading_day"] == 1]
    latest = daily_all.iloc[-1]
    forecast_frame = make_forecast(horizon)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total revenue (selected range)", gbp(view[config.TARGET_COLUMN].sum()))
    c2.metric("Average daily revenue", gbp(view[config.TARGET_COLUMN].mean()),
              help="Average over all calendar days in the range, including non-trading days.")
    c3.metric("Average per trading day", gbp(trading_view[config.TARGET_COLUMN].mean()) if len(trading_view) else "n/a")
    c4, c5, c6 = st.columns(3)
    c4.metric("Total transactions (invoices)", f"{int(view['n_invoices'].sum()):,}")
    c5.metric("Unique customers (whole period)", f"{summary['unique_customers']:,}",
              help="Unique customer IDs over the full period. A customer count for an arbitrary date range cannot be "
                   "derived from daily totals. About 22% of lines have no customer ID and are not counted.")
    c6.metric("Latest actual revenue", gbp(latest[config.TARGET_COLUMN]), help=f"Last full day: {latest[config.DATE_COLUMN].date()}")
    c7, c8, c9 = st.columns(3)
    c7.metric("Forecast horizon", f"{horizon} days")
    c8.metric(f"Forecast total (next {horizon} days)", gbp(forecast_frame["forecast"].sum()))
    c9.metric("Days with forecast revenue", f"{int((forecast_frame['forecast'] > 0).sum())} of {horizon}")

    st.subheader("Revenue at a glance")
    glance = aggregate(view, "Weekly")
    fig = go.Figure(go.Scatter(x=glance.index, y=glance.values, mode="lines", line=dict(color=COLORS["accent"], width=2)))
    fig.update_layout(title="Weekly revenue in the selected range")
    st.plotly_chart(style_figure(fig, 340), width="stretch")
    st.info("**How to read the numbers.** Revenue is gross sales of products (cancellations, postage/fees and same-day "
            "corrected orders are excluded and tracked separately). Saturdays, UK holidays and 24 Dec - 3 Jan have no sales.")

# ----------------------------------------------------------------------------
# 2. Historical sales
# ----------------------------------------------------------------------------
with tabs[1]:
    level = st.radio("Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True)
    series_view = aggregate(view, level)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series_view.index, y=series_view.values, name=f"{level} revenue",
                             line=dict(color=COLORS["actual"], width=1.4 if level == "Daily" else 2.2)))
    if level == "Daily":
        for window, color in ((7, COLORS["accent"]), (28, COLORS["forecast"])):
            fig.add_trace(go.Scatter(x=series_view.index, y=series_view.rolling(window).mean(), name=f"{window}-day average",
                                     line=dict(color=color, width=2)))
    elif level == "Weekly":
        fig.add_trace(go.Scatter(x=series_view.index, y=series_view.rolling(4).mean(), name="4-week average",
                                 line=dict(color=COLORS["forecast"], width=2)))
    fig.update_layout(title=f"{level} gross revenue")
    st.plotly_chart(style_figure(fig, 460), width="stretch")
    if level != "Daily":
        st.caption("The first and last periods can be partial if the date range does not start/end on a period boundary.")
    with st.expander("Show the data"):
        st.dataframe(series_view.rename("revenue").to_frame().style.format("£{:,.0f}"), width="stretch")

# ----------------------------------------------------------------------------
# 3. Sales analysis
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Seasonality")
    left, right = st.columns(2)
    trading_view = view[view["is_trading_day"] == 1].copy()
    if trading_view.empty:
        left.info("No trading days in the selected range.")
    else:
        by_weekday = trading_view.groupby(trading_view[config.DATE_COLUMN].dt.dayofweek)[config.TARGET_COLUMN].mean()
        fig = px.bar(x=[WEEKDAYS[i] for i in by_weekday.index], y=by_weekday.values,
                     title="Average revenue by weekday (trading days only)", color_discrete_sequence=[COLORS["accent"]])
        left.plotly_chart(style_figure(fig, 340), width="stretch")
        by_month = trading_view.groupby(trading_view[config.DATE_COLUMN].dt.month)[config.TARGET_COLUMN].mean()
        fig = px.bar(x=[MONTHS[i - 1] for i in by_month.index], y=by_month.values,
                     title="Average revenue per trading day by month", color_discrete_sequence=[COLORS["actual"]])
        right.plotly_chart(style_figure(fig, 340), width="stretch")
        st.caption("Saturday is effectively closed (it is omitted when it has no trading days). Month averages mix "
                   "different numbers of years, so compare them with care.")

    st.subheader("Countries")
    country_daily = load_country_daily()
    in_range = country_daily[(country_daily[config.DATE_COLUMN] >= start_date) & (country_daily[config.DATE_COLUMN] <= end_date)]
    totals = in_range.groupby("country")[config.TARGET_COLUMN].sum().sort_values(ascending=False)
    chosen = [c for c in selected_countries if c in totals.index]
    if not chosen:
        st.info("Select at least one country with sales in the chosen range (sidebar).")
    else:
        shown = totals.loc[chosen]
        share = shown / totals.sum() * 100
        fig = px.bar(x=shown.index, y=shown.values, title="Revenue by selected countries",
                     color_discrete_sequence=[COLORS["accent"]])
        st.plotly_chart(style_figure(fig, 340), width="stretch")
        st.dataframe(pd.DataFrame({"revenue": shown.round(0), "share of range revenue (%)": share.round(2)})
                     .style.format({"revenue": "£{:,.0f}", "share of range revenue (%)": "{:.2f}"}), width="stretch")
    uk_share = float(country_summary_all.loc[country_summary_all["country"] == "United Kingdom", "revenue_share_pct"].iloc[0])
    st.caption(f"Over the whole period, the United Kingdom accounts for {uk_share:.1f}% of revenue across "
               f"{len(country_summary_all)} countries.")

    st.subheader("Cancellations and returns (tracked separately, never netted into revenue)")
    monthly = (view.set_index(config.DATE_COLUMN)[[config.TARGET_COLUMN, "cancel_value", "cancel_invoices", "n_invoices"]]
               .resample("MS").sum())
    monthly = monthly[monthly[config.TARGET_COLUMN] > 0]
    if monthly.empty:
        st.info("No monthly data available for the selected range.")
    else:
        monthly["cancel_ratio_pct"] = monthly["cancel_value"] / monthly[config.TARGET_COLUMN] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["cancel_value"], name="Cancelled value", marker_color=COLORS["muted"]))
        fig.add_trace(go.Scatter(x=monthly.index, y=monthly["cancel_ratio_pct"], name="Cancelled value / gross revenue (%)",
                                 yaxis="y2", line=dict(color=COLORS["forecast"], width=2)))
        fig.update_layout(title="Monthly cancellations", yaxis2=dict(overlaying="y", side="right", ticksuffix="%", showgrid=False))
        st.plotly_chart(style_figure(fig, 360), width="stretch")
        st.caption("A single very large cancelled order can dominate a month's ratio (see the business insights).")

# ----------------------------------------------------------------------------
# 4. Forecast
# ----------------------------------------------------------------------------
with tabs[3]:
    forecast_frame = make_forecast(horizon)
    history_days = st.selectbox("History shown before the forecast", [60, 90, 180, 365], index=1)
    history = daily_all.set_index(config.DATE_COLUMN)[config.TARGET_COLUMN].iloc[-history_days:]
    start_x = (daily_all[config.DATE_COLUMN].iloc[-1] + pd.Timedelta(hours=12))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="Actual (history)", line=dict(color=COLORS["actual"], width=2)))
    fig.add_trace(go.Scatter(x=forecast_frame.index, y=forecast_frame["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast_frame.index, y=forecast_frame["lower"], fill="tonexty", fillcolor=COLORS["band"],
                             line=dict(width=0), name=f"{int(meta['prediction_interval']['nominal_coverage'] * 100)}% empirical interval"))
    fig.add_trace(go.Scatter(x=forecast_frame.index, y=forecast_frame["forecast"], name="Forecast", mode="lines+markers",
                             line=dict(color=COLORS["forecast"], width=2.5, dash="dot"), marker=dict(size=5)))
    fig.add_shape(type="line", x0=start_x, x1=start_x, y0=0, y1=1, yref="paper", line=dict(color="black", dash="dot", width=1))
    fig.add_annotation(x=start_x, y=1, yref="paper", text="forecast starts", showarrow=False, xanchor="left", yanchor="top")
    fig.update_layout(title=f"Actual revenue and {horizon}-day forecast")
    st.plotly_chart(style_figure(fig, 480), width="stretch")

    table = forecast_frame.copy()
    table.insert(0, "weekday", table.index.day_name())
    table.index.name = "date"
    st.dataframe(table.rename(columns={"forecast": "forecast (£)", "lower": "lower (£)", "upper": "upper (£)"})
                 .style.format({"forecast (£)": "£{:,.0f}", "lower (£)": "£{:,.0f}", "upper (£)": "£{:,.0f}"}),
                 width="stretch", height=360)
    st.download_button("Download forecast as CSV", table.to_csv().encode("utf-8"), file_name=f"forecast_{horizon}d.csv", mime="text/csv")

    st.warning(
        "**Read this forecast with care.** It is a model estimate, not a guarantee. Days known in advance to be closed "
        "(UK holidays, 24 Dec - 3 Jan) are forecast as zero. The interval is an *empirical* band built from validation "
        f"errors (observed test coverage {meta['prediction_interval']['observed_test_coverage'] * 100:.0f}% on 30 days); it is "
        "an approximation. The model has seen only one previous December, and its mid-December level is high compared "
        "with the same dates a year earlier (see notebook 05).")

# ----------------------------------------------------------------------------
# 5. Model performance
# ----------------------------------------------------------------------------
with tabs[4]:
    comparison = load_comparison()
    st.subheader("Final model")
    a, b, c = st.columns(3)
    a.metric("Selected model", meta["model_name"])
    test_rec = meta["test_metrics_pool_trained_model"]["30-day recursive"]
    b.metric("Test MAE (30-day recursive)", gbp(test_rec["mae"]))
    c.metric("Test WAPE", f"{test_rec['wape']:.1f}%")
    st.caption(meta["test_note"])

    protocol = st.radio("Evaluation protocol", ["30-day recursive", "1-day-ahead"], horizontal=True,
                        help="30-day recursive: forecasts feed later lags (how the deployed forecast works). "
                             "1-day-ahead: each day is predicted from actuals up to the previous day.")
    metric_cols = ["model", "kind", "mae", "rmse", "smape_trading_days", "wape", "r2"]
    rename = {"mae": "MAE (£)", "rmse": "RMSE (£)", "smape_trading_days": "sMAPE trading days (%)", "wape": "WAPE (%)", "r2": "R²"}
    fmt = {"MAE (£)": "{:,.0f}", "RMSE (£)": "{:,.0f}", "sMAPE trading days (%)": "{:.1f}", "WAPE (%)": "{:.1f}", "R²": "{:.2f}"}
    for label, prefix in (("Validation (mean of 4 chronological folds, model selection)", "validation"),
                          ("Final test set (last 30 days, evaluated once)", "test")):
        st.markdown(f"**{label}**")
        part = comparison[(comparison["split"].str.startswith(prefix)) & (comparison["protocol"] == protocol)]
        st.dataframe(part.sort_values("mae")[metric_cols].rename(columns=rename).style.format(fmt),
                     width="stretch", hide_index=True)

    chart_data = comparison[(comparison["protocol"] == protocol)].copy()
    chart_data["split_label"] = chart_data["split"].str.split(" ").str[0]
    fig = px.bar(chart_data, x="model", y="mae", color="split_label", barmode="group", title=f"MAE by model ({protocol})",
                 color_discrete_map={"validation": COLORS["muted"], "test": COLORS["forecast"]})
    fig.update_layout(xaxis_title=None, xaxis_tickangle=-25)
    st.plotly_chart(style_figure(fig, 440), width="stretch")
    st.caption("Baselines on the test split are shown for reference only; they have no fitted parameters and "
               "played no part in selecting the model. Other candidate models were never scored on the test set.")

    test_pred = load_test_predictions().set_index(config.DATE_COLUMN)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred["lower"], fill="tonexty", fillcolor=COLORS["band"], line=dict(width=0), name="80% interval"))
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred["actual"], name="Actual", line=dict(color=COLORS["actual"], width=2)))
    fig.add_trace(go.Scatter(x=test_pred.index, y=test_pred["forecast"], name="Forecast", line=dict(color=COLORS["forecast"], width=2)))
    fig.update_layout(title="Final test period: actual vs forecast (model trained only on earlier data)")
    st.plotly_chart(style_figure(fig, 400), width="stretch")

    with st.expander("Model details and reproducibility"):
        st.json({"model": meta["model_name"], "hyperparameters": meta["hyperparameters"],
                 "selection_rule": meta["selection_rule"], "features": meta["feature_columns"],
                 "saved_model_training_period": meta["saved_model_training_period"], "versions": meta["versions"]})

# ----------------------------------------------------------------------------
# 6. Business insights
# ----------------------------------------------------------------------------
with tabs[5]:
    st.markdown(load_insights_text())
