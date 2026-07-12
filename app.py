"""Streamlit dashboard: Large-Scale Analysis of COVID-19 Epidemiological Data

UI follows a deep dark Palantir-like aesthetic. Uses PySpark (local[*]) for heavy lifting
and Plotly for interactive charts. Features an interactive world map showing COVID cases by country.
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


st.set_page_config(page_title="COVID-19 Global Analysis", layout="wide")


CSS = """
<style>
html, body, .main { background: #11151c; color: #e2e8f0; }
.stApp { background: #11151c; }
.card { background: #1a202c; border: 1px solid #2d3748; padding: 14px; border-radius: 6px; }
.metric-card { display: inline-block; padding: 16px 18px; margin: 6px; border-radius: 6px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.04)); box-shadow: 0 2px 12px rgba(0,0,0,0.6); }
.metric-value { font-size: 24px; font-weight: 700; color: #e2e8f0; }
.metric-label { font-size: 12px; color: #cbd5e1; }
.glow { box-shadow: 0 0 12px rgba(49,130,206,0.12); border: 1px solid rgba(49,130,206,0.18); }
.top-metrics { display:flex; gap:12px; align-items:center; flex-wrap: wrap; }
.risk-high { color: #ff4d4f; font-weight:800; font-size:20px; }
.risk-med { color: #ffb020; font-weight:700; font-size:18px; }
.risk-low { color: #48bb78; font-weight:700; font-size:18px; }
.country-card { background: #1a202c; border: 1px solid #2d3748; padding: 16px; border-radius: 6px; margin: 8px 0; cursor: pointer; transition: all 0.2s; }
.country-card:hover { border-color: #3182ce; background: #1d2639; }
.country-name { font-size: 16px; font-weight: 600; color: #e2e8f0; }
.country-stat { font-size: 13px; color: #cbd5e1; margin: 4px 0; }

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
def load_data(days=180, regions=None):
    """Initialize Spark, generate dataset and compute aggregated tables."""
    if regions is None:
        regions = len(data_processor.COUNTRIES_DATA)
    
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
                .groupBy("Country", "month")
                .agg(F.sum("Confirmed").alias("monthly_confirmed"))
                .orderBy("Country", "month")
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
        month_df = month_df.groupby(["Country", "month"]).agg({"Confirmed": "sum"}).reset_index().rename(columns={"Confirmed": "monthly_confirmed"})

    return sdf, summary, ts, region_df, month_df


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown("<h1 style='margin:0 0 12px 0;color:#e2e8f0;text-align:center;font-size:28px'>COVID-19 Global Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#cbd5e1;margin-bottom:20px;font-size:13px'>Interactive map • Geographically accurate • Real-time statistics</p>", unsafe_allow_html=True)

    # Load data (cached)
    with st.spinner("Loading global COVID-19 data..."):
        sdf, summary, ts, region_df, month_df = load_data(days=180)

    # Top metric bar
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns([1.5, 1.5, 1.5, 1.5])

    # Determine risk level by active rate
    active_rate = summary.get("active_rate", 0.0)
    if active_rate > 0.12:
        risk_html = "<span class='risk-high'>HIGH</span>"
    elif active_rate > 0.05:
        risk_html = "<span class='risk-med'>MEDIUM</span>"
    else:
        risk_html = "<span class='risk-low'>LOW</span>"

    metric_col1.markdown(f"<div class='card metric-card glow'><div class='metric-label'>Risk Level</div><div style='margin-top:6px'>{risk_html}</div></div>", unsafe_allow_html=True)
    metric_col2.markdown(f"<div class='card metric-card'><div class='metric-label'>Global Cases</div><div class='metric-value'>{human_format(summary.get('total_confirmed', 0))}</div></div>", unsafe_allow_html=True)
    metric_col3.markdown(f"<div class='card metric-card'><div class='metric-label'>Deaths</div><div class='metric-value'>{human_format(summary.get('total_deaths', 0))}</div></div>", unsafe_allow_html=True)
    metric_col4.markdown(f"<div class='card metric-card'><div class='metric-label'>Recovered</div><div class='metric-value'>{human_format(summary.get('total_recoveries', 0))}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d3748;margin:20px 0'/>", unsafe_allow_html=True)

    # Main content: Interactive World Map + Country Search
    map_col, search_col = st.columns([3, 1])

    with map_col:
        st.markdown("<div class='card' style='padding:0;overflow:hidden'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:14px 14px 6px 14px;color:#e2e8f0'>Interactive World Map</h4>", unsafe_allow_html=True)

        if not region_df.empty:
            # Create interactive world map using scatter_geo
            fig_map = px.scatter_geo(
                region_df,
                lat='Latitude',
                lon='Longitude',
                size='confirmed',
                color='active_per_100k',
                hover_name='Country',
                hover_data={'Latitude': ':.2f', 'Longitude': ':.2f', 'confirmed': ':,', 'active_per_100k': ':.1f'},
                title='',
                size_max=50,
                color_continuous_scale='RdYlGn_r',
                template='plotly_dark'
            )
            
            fig_map.update_layout(
                geo=dict(
                    projection_type='natural earth',
                    bgcolor='#0f1117',
                    coastlinecolor='#2d3748',
                    landcolor='#1a202c',
                    showland=True,
                    countrycolor='#2d3748',
                    countrywidth=0.5,
                    framecolor='#2d3748',
                    framewidth=1,
                ),
                paper_bgcolor='#11151c',
                plot_bgcolor='#11151c',
                font=dict(color='#e2e8f0', size=11),
                margin=dict(r=0, t=0, l=0, b=0),
                height=600,
                hovermode='closest',
                coloraxis_colorbar=dict(
                    title="Active<br>Per 100k",
                    tickcolor="#e2e8f0",
                    tickfont={"color": "#e2e8f0", "size": 10},
                    title_font={"color": "#e2e8f0", "size": 11}
                )
            )
            
            fig_map.update_traces(marker=dict(line=dict(width=1, color="#11151c")))
            
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No map data available")

        st.markdown("</div>", unsafe_allow_html=True)

    with search_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:6px 0 12px 0;color:#e2e8f0;font-size:14px'>Search Country</h4>", unsafe_allow_html=True)
        
        # Get list of countries with data
        countries_list = sorted(region_df['Country'].unique().tolist()) if not region_df.empty else []
        
        selected_country = st.selectbox(
            "Select a country to view details",
            countries_list,
            label_visibility="collapsed",
            key="country_search"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d3748;margin:20px 0'/>", unsafe_allow_html=True)

    # Country Details Section
    if selected_country and not region_df.empty:
        country_data = region_df[region_df['Country'] == selected_country]
        
        if not country_data.empty:
            row = country_data.iloc[0]
            
            st.markdown(f"<h3 style='color:#e2e8f0;margin-bottom:16px'>{selected_country} - Detailed Statistics</h3>", unsafe_allow_html=True)
            
            # Country metrics
            country_metric_col1, country_metric_col2, country_metric_col3, country_metric_col4 = st.columns(4)
            
            country_metric_col1.markdown(f"<div class='card metric-card'><div class='metric-label'>Total Cases</div><div class='metric-value'>{int(row['confirmed']):,}</div></div>", unsafe_allow_html=True)
            country_metric_col2.markdown(f"<div class='card metric-card'><div class='metric-label'>Active Per 100k</div><div class='metric-value'>{row['active_per_100k']:.1f}</div></div>", unsafe_allow_html=True)
            country_metric_col3.markdown(f"<div class='card metric-card'><div class='metric-label'>Latitude</div><div class='metric-value'>{row['Latitude']:.2f}°</div></div>", unsafe_allow_html=True)
            country_metric_col4.markdown(f"<div class='card metric-card'><div class='metric-label'>Longitude</div><div class='metric-value'>{row['Longitude']:.2f}°</div></div>", unsafe_allow_html=True)
            
            # Get country time series data
            country_pdf = data_processor.get_mock_data(spark=None, days=180, regions=len(data_processor.COUNTRIES_DATA))[1]
            country_ts = country_pdf[country_pdf['Country'] == selected_country].sort_values('Date')
            
            if not country_ts.empty:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin:6px 0;color:#e2e8f0'>Cases Timeline</h4>", unsafe_allow_html=True)
                
                fig_country = make_subplots(specs=[[{"secondary_y": True}]])
                fig_country.add_trace(
                    go.Bar(x=country_ts["Date"], y=country_ts["Confirmed"], name="Daily Cases", marker_color="#3182ce"),
                    secondary_y=False
                )
                
                country_ts['cumulative'] = country_ts['Confirmed'].cumsum()
                fig_country.add_trace(
                    go.Scatter(x=country_ts["Date"], y=country_ts['cumulative'], name="Cumulative", 
                               line=dict(color="#e2e8f0", width=2), mode='lines'),
                    secondary_y=True
                )
                
                fig_country.update_layout(
                    template="plotly_dark",
                    bargap=0.1,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor="#11151c",
                    plot_bgcolor="#1a202c",
                    font=dict(color="#e2e8f0"),
                    margin=dict(r=0, t=0, l=0, b=0),
                    height=300
                )
                fig_country.update_yaxes(title_text="Daily Cases", secondary_y=False, title_font=dict(color="#e2e8f0"))
                fig_country.update_yaxes(title_text="Cumulative Cases", secondary_y=True, title_font=dict(color="#e2e8f0"))
                
                st.plotly_chart(fig_country, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

    # Global trends
    st.markdown("<hr style='border-color:#2d3748;margin:20px 0'/>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#e2e8f0;margin-bottom:16px'>Global Trends</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:6px 0;color:#e2e8f0'>Cases Over Time (Global)</h4>", unsafe_allow_html=True)
        
        if not ts.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=ts["Date"], y=ts["daily_confirmed"], name="Daily Cases", marker_color="#3182ce"), secondary_y=False)
            fig.add_trace(go.Line(x=ts["Date"], y=ts["cumulative_confirmed"], name="Cumulative", line=dict(color="#e2e8f0", width=2)), secondary_y=True)
            fig.update_layout(template="plotly_dark", bargap=0.1, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
                            paper_bgcolor="#11151c", plot_bgcolor="#1a202c", font=dict(color="#e2e8f0"), height=350)
            fig.update_yaxes(title_text="Daily Cases", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Cases", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:6px 0;color:#e2e8f0'>Top 10 Countries by Cases</h4>", unsafe_allow_html=True)
        
        if not region_df.empty:
            top_countries = region_df.nlargest(10, 'confirmed')[['Country', 'confirmed', 'active_per_100k']]
            
            fig_top = go.Figure()
            fig_top.add_trace(go.Bar(
                y=top_countries['Country'],
                x=top_countries['confirmed'],
                orientation='h',
                marker_color='#3182ce',
                text=[f"{int(x):,}" for x in top_countries['confirmed']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Cases: %{x:,.0f}<extra></extra>'
            ))
            
            fig_top.update_layout(
                template="plotly_dark",
                paper_bgcolor="#11151c",
                plot_bgcolor="#1a202c",
                font=dict(color="#e2e8f0"),
                margin=dict(l=120, r=0, t=0, b=0),
                height=350,
                xaxis_title="",
                yaxis_title="",
                showlegend=False
            )
            fig_top.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#2d3748")
            
            st.plotly_chart(fig_top, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
