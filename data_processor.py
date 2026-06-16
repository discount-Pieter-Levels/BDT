"""data_processor.py - Robust COVID-19 data generation with Spark/pandas fallback."""
import random
import math
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Safe PySpark imports
SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
    SPARK_AVAILABLE = True
except Exception:
    pass


def init_spark(app_name="covid_analysis_local"):
    """Try to init Spark; return None if unavailable."""
    if not SPARK_AVAILABLE:
        return None
    try:
        spark = (
            SparkSession.builder.master("local[*]")
            .appName(app_name)
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        return spark
    except Exception:
        return None


def get_mock_data(spark=None, days=365, regions=200):
    """
    Generate synthetic COVID-19-like dataset. Always returns (spark_df, pandas_df) tuple.
    
    If spark is None or unavailable, spark_df will be None and pandas_df contains the data.
    Columns: Date, Region, Latitude, Longitude, Confirmed, Deaths, Recoveries, Active_Per_100k, Retail_Sales_Impact
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days - 1)

    # Create a list of synthetic regions with approximate lat/lon
    region_list = []
    for i in range(regions):
        name = f"Region-{i+1}"
        # sample lat/lon roughly across the globe
        lat = random.uniform(-55, 70)
        lon = random.uniform(-170, 170)
        pop = random.randint(50_000, 50_000_00)  # population proxy
        region_list.append((name, lat, lon, pop))

    rows = []
    for rname, lat, lon, pop in region_list:
        baseline_cases = random.randint(1000, 100_000)
        baseline_deaths = int(baseline_cases * random.uniform(0.005, 0.03))
        baseline_recoveries = int(baseline_cases * random.uniform(0.6, 0.95))

        for day_idx in range(days):
            date = start_date + timedelta(days=day_idx)

            # simple epidemic curve noise + seasonality
            seasonal = 1 + 0.25 * (1 + math_sine(day_idx / 30.0))
            growth = 1 + (day_idx / (days * 10.0))
            confirmed = max(0, int(baseline_cases * seasonal * growth * random.uniform(0.8, 1.2)))
            deaths = max(0, int(confirmed * random.uniform(0.001, 0.03)))
            recoveries = max(0, int(confirmed * random.uniform(0.5, 0.95)))

            active_per_100k = (confirmed - recoveries) / max(1, (pop / 100_000))
            retail_impact = round(random.uniform(-0.25, 0.15), 3)  # negative means drop

            rows.append((date, rname, float(lat), float(lon), int(confirmed), int(deaths), int(recoveries), float(active_per_100k), float(retail_impact)))

    pdf = pd.DataFrame(rows, columns=["Date", "Region", "Latitude", "Longitude", "Confirmed", "Deaths", "Recoveries", "Active_Per_100k", "Retail_Sales_Impact"]) 

    # Try to convert to Spark if available
    sdf = None
    if spark is not None:
        try:
            schema = StructType([
                StructField("Date", DateType(), False),
                StructField("Region", StringType(), False),
                StructField("Latitude", DoubleType(), False),
                StructField("Longitude", DoubleType(), False),
                StructField("Confirmed", IntegerType(), False),
                StructField("Deaths", IntegerType(), False),
                StructField("Recoveries", IntegerType(), False),
                StructField("Active_Per_100k", DoubleType(), False),
                StructField("Retail_Sales_Impact", DoubleType(), False),
            ])
            sdf = spark.createDataFrame(pdf, schema=schema)
            sdf = sdf.repartition(4)
            sdf.cache()
            sdf.count()  # materialize cache for snappy response in the app
        except Exception:
            sdf = None
    
    return sdf, pdf


def math_sine(x):
    # tiny helper to avoid importing math globally in row loops
    import math

    return math.sin(x)


def compute_global_summary(sdf):
    """Compute global summary metrics using PySpark transformations."""
    total_confirmed = sdf.agg(F.sum("Confirmed").alias("total_confirmed")).collect()[0]["total_confirmed"]
    total_deaths = sdf.agg(F.sum("Deaths").alias("total_deaths")).collect()[0]["total_deaths"]
    total_recoveries = sdf.agg(F.sum("Recoveries").alias("total_recoveries")).collect()[0]["total_recoveries"]

    # Active approximated as confirmed - recoveries
    total_active = total_confirmed - total_recoveries
    recovery_rate = (total_recoveries / total_confirmed) if total_confirmed else 0
    active_rate = (total_active / total_confirmed) if total_confirmed else 0

    return {
        "total_confirmed": int(total_confirmed),
        "total_deaths": int(total_deaths),
        "total_recoveries": int(total_recoveries),
        "total_active": int(total_active),
        "recovery_rate": float(recovery_rate),
        "active_rate": float(active_rate),
    }


def compute_time_series(sdf):
    """Return a Pandas DataFrame with daily aggregated metrics for plotting."""
    ts = (
        sdf.groupBy("Date")
        .agg(
            F.sum("Confirmed").alias("daily_confirmed"),
            F.sum("Deaths").alias("daily_deaths"),
            F.sum("Recoveries").alias("daily_recoveries"),
        )
        .orderBy("Date")
    )
    pdf = ts.toPandas()
    pdf["cumulative_confirmed"] = pdf["daily_confirmed"].cumsum()
    return pdf


def compute_region_aggregations(sdf):
    """Aggregate by region to get recent snapshots for mapping and tables."""
    # Use latest date per region
    window_max = sdf.groupBy("Region").agg(F.max("Date").alias("latest_date"))
    joined = sdf.join(window_max, (sdf.Region == window_max.Region) & (sdf.Date == window_max.latest_date), how="inner").select(sdf["*"])

    regions = (
        joined.groupBy("Region", "Latitude", "Longitude")
        .agg(
            F.sum("Confirmed").alias("confirmed"),
            F.sum("Deaths").alias("deaths"),
            F.sum("Recoveries").alias("recoveries"),
            F.avg("Active_Per_100k").alias("active_per_100k"),
        )
        .orderBy(F.desc("confirmed"))
    )

    return regions.toPandas()


if __name__ == "__main__":
    # Test with pandas fallback (Spark may fail to init)
    print("Testing data processor with pandas fallback...")
    sdf, pdf = get_mock_data(spark=None, days=30, regions=10)
    summary = compute_global_summary_from_pdf(pdf)
    print(f"✓ Generated {len(pdf)} rows, {len(pdf['Region'].unique())} regions")
    print(f"✓ Global summary: {summary['total_cases']} cases, {summary['recovery_rate']:.1%} recovery rate")


# --- Pandas fallback implementations ---
def compute_global_summary_from_pdf(pdf: pd.DataFrame):
    """Compute global summary metrics from a pandas DataFrame (fallback when Spark unavailable)."""
    # take latest record per region
    latest = pdf.sort_values(["Region", "Date"]).groupby("Region").tail(1)

    total_cases = int(latest["Confirmed"].sum())
    total_deaths = int(latest["Deaths"].sum())
    total_recoveries = int(latest["Recoveries"].sum()) if "Recoveries" in latest.columns else 0
    total_active = int(latest.get("Active", (latest["Confirmed"] - latest.get("Recoveries", 0))).sum())

    recovery_rate = float(total_recoveries) / total_cases if total_cases else 0.0
    active_rate = float(total_active) / total_cases if total_cases else 0.0

    return {
        "total_cases": total_cases,
        "total_deaths": total_deaths,
        "total_recoveries": total_recoveries,
        "total_active": total_active,
        "recovery_rate": recovery_rate,
        "active_rate": active_rate,
    }


def compute_time_series_from_pdf(pdf: pd.DataFrame):
    """Aggregate pandas DataFrame into daily time series for plotting."""
    daily = pdf.groupby("Date").agg({
        "Confirmed": "sum",
        "Deaths": "sum",
        "Recoveries": "sum",
    }).reset_index().rename(columns={"Confirmed": "daily_confirmed", "Deaths": "daily_deaths", "Recoveries": "daily_recoveries"})
    daily = daily.sort_values("Date")
    daily["cumulative_confirmed"] = daily["daily_confirmed"].cumsum()
    return daily


def compute_region_aggregations_from_pdf(pdf: pd.DataFrame):
    """Produce latest snapshot aggregated by region from pandas DataFrame for mapping and tables."""
    latest = pdf.sort_values(["Region", "Date"]).groupby("Region").tail(1)
    out = latest[["Region", "Latitude", "Longitude", "Confirmed", "Recoveries", "Active_Per_100k"]].copy()
    out["confirmed"] = out["Confirmed"]
    out["active_per_100k"] = out["Active_Per_100k"]
    return out[["Region", "Latitude", "Longitude", "confirmed", "active_per_100k"]]
