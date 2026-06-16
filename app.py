"""Streamlit dashboard: Large-Scale Analysis of COVID-19 Epidemiological Data

UI follows a deep dark Palantir-like aesthetic. Uses PySpark (local[*]) for heavy lifting
and Plotly for interactive charts. The app imports `data_processor` to generate and
aggregate a synthetic COVID dataset so the app runs immediately without external files.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data_processor

# PySpark imports - only used if Spark is available
try:
    from pyspark.sql import functions as F
except Exception:
    F = None


st.set_page_config(page_title="Large-Scale COVID-19 Analysis", layout="wide")


CSS = """
<style>
html, body, .main { background: #11151c; color: #e2e8f0; }
.stApp { background: #11151c; }
.card { background: #1a202c; border: 1px solid #2d3748; padding: 14px; border-radius: 6px; }
.metric-card { display: inline-block; padding: 16px 18px; margin: 6px; border-radius: 6px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.04)); box-shadow: 0 2px 12px rgba(0,0,0,0.6); }
.metric-value { font-size: 24px; font-weight: 700; color: #e2e8f0; }
.metric-label { font-size: 12px; color: #cbd5e1; }
.glow { box-shadow: 0 0 12px rgba(49,130,206,0.12); border: 1px solid rgba(49,130,206,0.18); }
.top-metrics { display:flex; gap:12px; align-items:center; }
.risk-high { color: #ff4d4f; font-weight:800; font-size:20px; }
.risk-med { color: #ffb020; font-weight:700; font-size:18px; }
.risk-low { color: #48bb78; font-weight:700; font-size:18px; }

/* make tables and streamlit components respect dark theme */
div.block-container { padding-top: 1rem; }
</style>
"""


def human_format(num):
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"


@st.cache_data(ttl=600)
def load_data(days=180, regions=120):
    """Initialize Spark, generate dataset and compute aggregated tables."""
    spark = data_processor.init_spark()

    # get_mock_data may return either a Spark DataFrame or (spark_df, pandas_df)
    result = data_processor.get_mock_data(spark=spark, days=days, regions=regions)

    # normalize outputs
    sdf = None
    pdf = None
    if isinstance(result, tuple) and len(result) == 2:
        sdf, pdf = result
    else:
        sdf = result

    # If Spark started successfully and sdf is available, use PySpark aggregations
    if sdf is not None and F is not None:
        try:
            summary = data_processor.compute_global_summary(sdf)
            ts = data_processor.compute_time_series(sdf)
            region_df = data_processor.compute_region_aggregations(sdf)
            month_df = (
                sdf.withColumn("month", F.date_format(F.col("Date"), "yyyy-MM"))
                .groupBy("Region", "month")
                .agg(F.sum("Confirmed").alias("monthly_confirmed"))
                .orderBy("Region", "month")
                .toPandas()
            )
        except Exception:
            # If Spark operations fail, fall back to pandas
            sdf = None
    
    if sdf is None:
        # Fall back to pandas-only operations (use pdf)
        if pdf is None:
            # as a last resort regenerate pandas data
            _, pdf = data_processor.get_mock_data(spark=None, days=days, regions=regions)

        summary = data_processor.compute_global_summary_from_pdf(pdf)
        ts = data_processor.compute_time_series_from_pdf(pdf)
        region_df = data_processor.compute_region_aggregations_from_pdf(pdf)
        # monthly grouping
        month_df = pdf.copy()
        month_df["month"] = pd.to_datetime(month_df["Date"]).dt.to_period("M").astype(str)
        month_df = month_df.groupby(["Region", "month"]).agg({"Confirmed": "sum"}).reset_index().rename(columns={"Confirmed": "monthly_confirmed"})

    return sdf, summary, ts, region_df, month_df


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'>"
                "<h2 style='margin:0;color:#e2e8f0'>Large-Scale Analysis of COVID-19 Epidemiological Data</h2>"
                "</div>", unsafe_allow_html=True)

    # Load data (cached)
    with st.spinner("Generating synthetic dataset and computing aggregates (PySpark)…"):
        sdf, summary, ts, region_df, month_df = load_data(days=180, regions=120)

    # Top metric bar
    top_col1, top_col2, top_col3, top_col4 = st.columns([2, 1.6, 1.6, 1.6])

    # Determine risk level by active rate
    active_rate = summary.get("active_rate", 0.0)
    if active_rate > 0.12:
        risk_html = "<span class='risk-high'>HIGH</span>"
    elif active_rate > 0.05:
        risk_html = "<span class='risk-med'>MEDIUM</span>"
    else:
        risk_html = "<span class='risk-low'>LOW</span>"

    top_col1.markdown(f"<div class='card metric-card glow'><div class='metric-label'>Overall Risk Level</div><div style='margin-top:6px'>{risk_html}</div></div>", unsafe_allow_html=True)

    delta = None
    try:
        if not ts.empty and "cumulative_confirmed" in ts.columns:
            last = int(ts.iloc[-1]["cumulative_confirmed"])
            prev = int(ts.iloc[-2]["cumulative_confirmed"]) if len(ts) > 1 else 0
            delta = last - prev
    except Exception:
        delta = None

    delta_html = f"<span style='color:#48bb78'>&uarr; {delta}</span>" if delta and delta >= 0 else f"<span style='color:#ff4d4f'>{delta}</span>"

    top_col2.markdown(f"<div class='card metric-card'><div class='metric-label'>Total Cases</div><div class='metric-value'>{human_format(summary.get('total_confirmed', 0))} {delta_html if delta is not None else ''}</div></div>", unsafe_allow_html=True)
    top_col3.markdown(f"<div class='card metric-card'><div class='metric-label'>Active Cases Per 100k</div><div class='metric-value'>{summary.get('active_rate', 0.0)*100000:,.1f}</div></div>", unsafe_allow_html=True)
    top_col4.markdown(f"<div class='card metric-card'><div class='metric-label'>Est. Recoveries</div><div class='metric-value'>{human_format(summary.get('total_recoveries', 0))}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d3748'/>", unsafe_allow_html=True)

    # Main layout: left map, right tabs
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:6px 0;color:#e2e8f0'>Global Case Density Map</h4>", unsafe_allow_html=True)

        if not region_df.empty:
            fig_map = px.scatter_mapbox(
                region_df,
                lat="Latitude",
                lon="Longitude",
                size="confirmed",
                color="active_per_100k",
                hover_name="Region",
                size_max=30,
                color_continuous_scale="RdYlGn_r",
                template="plotly_dark",
                zoom=1,
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="#11151c", plot_bgcolor="#11151c")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No region data available")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        tabs = st.tabs(["COVID Charts", "Regional Data"])

        with tabs[0]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin:6px 0;color:#e2e8f0'>Cases Over Time</h4>", unsafe_allow_html=True)

            if not ts.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=ts["Date"], y=ts["daily_confirmed"], name="Daily Cases", marker_color="#3182ce"), secondary_y=False)
                fig.add_trace(go.Line(x=ts["Date"], y=ts["cumulative_confirmed"], name="Cumulative", line=dict(color="#e2e8f0", width=2)), secondary_y=True)
                fig.update_layout(template="plotly_dark", bargap=0.1, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig.update_yaxes(title_text="Daily Cases", secondary_y=False)
                fig.update_yaxes(title_text="Cumulative Cases", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time series available")

            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='margin:6px 0;color:#e2e8f0'>Regional Monthly Breakdown</h4>", unsafe_allow_html=True)

            if not month_df.empty:
                # pivot latest 3 months for readability
                month_summary = month_df.pivot(index="Region", columns="month", values="monthly_confirmed").fillna(0)
                # compute pct growth last vs previous
                if month_summary.shape[1] >= 2:
                    cols = list(month_summary.columns)
                    month_summary["pct_growth_last"] = ((month_summary[cols[-1]] - month_summary[cols[-2]]) / month_summary[cols[-2]].replace(0, 1)) * 100
                st.dataframe(month_summary.style.format(na_rep="-"))
            else:
                st.info("No monthly data")

            st.markdown("</div>", unsafe_allow_html=True)

    #st.markdown("<div style='padding:8px;color:#94a3b8'>Data generated synthetically via PySpark for demonstration purposes.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
