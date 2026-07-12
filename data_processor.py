"""data_processor.py - Robust COVID-19 data generation with real country data."""
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


# Real country data with ISO codes, centers, and populations
COUNTRIES_DATA = {
    "India": {"iso_alpha": "IND", "lat": 20.5937, "lon": 78.9629, "population": 1417173173},
    "China": {"iso_alpha": "CHN", "lat": 35.8617, "lon": 104.1954, "population": 1425887337},
    "United States": {"iso_alpha": "USA", "lat": 37.0902, "lon": -95.7129, "population": 339996563},
    "Indonesia": {"iso_alpha": "IDN", "lat": -0.7893, "lon": 113.9213, "population": 277534122},
    "Pakistan": {"iso_alpha": "PAK", "lat": 30.3753, "lon": 69.3451, "population": 240485658},
    "Brazil": {"iso_alpha": "BRA", "lat": -14.2350, "lon": -51.9253, "population": 215454994},
    "Nigeria": {"iso_alpha": "NGA", "lat": 9.0820, "lon": 8.6753, "population": 223000000},
    "Bangladesh": {"iso_alpha": "BGD", "lat": 23.6850, "lon": 90.3563, "population": 170993755},
    "Russia": {"iso_alpha": "RUS", "lat": 61.5240, "lon": 105.3188, "population": 144713314},
    "Mexico": {"iso_alpha": "MEX", "lat": 23.6345, "lon": -102.5528, "population": 128455507},
    "Japan": {"iso_alpha": "JPN", "lat": 36.2048, "lon": 138.2529, "population": 123294513},
    "Ethiopia": {"iso_alpha": "ETH", "lat": 9.1450, "lon": 40.4897, "population": 120283026},
    "Philippines": {"iso_alpha": "PHL", "lat": 12.8797, "lon": 121.7740, "population": 119106242},
    "Egypt": {"iso_alpha": "EGY", "lat": 26.8206, "lon": 30.8025, "population": 110887041},
    "Vietnam": {"iso_alpha": "VNM", "lat": 14.0583, "lon": 108.2772, "population": 97328541},
    "Democratic Republic of the Congo": {"iso_alpha": "COD", "lat": -4.0383, "lon": 21.7587, "population": 99010212},
    "Turkey": {"iso_alpha": "TUR", "lat": 38.9637, "lon": 35.2433, "population": 85326000},
    "Iran": {"iso_alpha": "IRN", "lat": 32.4279, "lon": 53.6880, "population": 90550570},
    "Germany": {"iso_alpha": "DEU", "lat": 51.1657, "lon": 10.4515, "population": 83368221},
    "Thailand": {"iso_alpha": "THA", "lat": 15.8700, "lon": 100.9925, "population": 71800000},
    "United Kingdom": {"iso_alpha": "GBR", "lat": 55.3781, "lon": -3.4360, "population": 67736802},
    "Tanzania": {"iso_alpha": "TZA", "lat": -6.3690, "lon": 34.8888, "population": 60142438},
    "France": {"iso_alpha": "FRA", "lat": 46.2276, "lon": 2.2137, "population": 68014000},
    "South Africa": {"iso_alpha": "ZAF", "lat": -30.5595, "lon": 22.9375, "population": 60142191},
    "Italy": {"iso_alpha": "ITA", "lat": 41.8719, "lon": 12.5674, "population": 59110627},
    "Kenya": {"iso_alpha": "KEN", "lat": -0.0236, "lon": 37.9062, "population": 54037487},
    "Myanmar": {"iso_alpha": "MMR", "lat": 21.9162, "lon": 95.9560, "population": 54409800},
    "Colombia": {"iso_alpha": "COL", "lat": 4.5709, "lon": -74.2973, "population": 51874024},
    "Sudan": {"iso_alpha": "SDN", "lat": 12.8628, "lon": 30.8025, "population": 45657202},
    "Ukraine": {"iso_alpha": "UKR", "lat": 48.3794, "lon": 31.1656, "population": 38000000},
    "Canada": {"iso_alpha": "CAN", "lat": 56.1304, "lon": -106.3468, "population": 39674209},
    "Algeria": {"iso_alpha": "DZA", "lat": 28.0339, "lon": 1.6596, "population": 44615272},
    "Argentina": {"iso_alpha": "ARG", "lat": -38.4161, "lon": -63.6167, "population": 46044703},
    "Iraq": {"iso_alpha": "IRQ", "lat": 33.2232, "lon": 43.6793, "population": 44496122},
    "Afghanistan": {"iso_alpha": "AFG", "lat": 33.9391, "lon": 67.7100, "population": 42235775},
    "Morocco": {"iso_alpha": "MAR", "lat": 31.7917, "lon": -7.0926, "population": 37344795},
    "Saudi Arabia": {"iso_alpha": "SAU", "lat": 23.8859, "lon": 45.0792, "population": 36408820},
    "Uzbekistan": {"iso_alpha": "UZB", "lat": 41.3775, "lon": 64.5853, "population": 35163944},
    "Angola": {"iso_alpha": "AGO", "lat": -11.2027, "lon": 17.8739, "population": 32866268},
    "Peru": {"iso_alpha": "PER", "lat": -9.1900, "lon": -75.0152, "population": 34354719},
    "Malaysia": {"iso_alpha": "MYS", "lat": 4.2105, "lon": 101.6964, "population": 34002200},
    "Ghana": {"iso_alpha": "GHA", "lat": 7.3697, "lon": -5.7418, "population": 34121985},
    "Yemen": {"iso_alpha": "YEM", "lat": 15.5527, "lon": 48.5164, "population": 34628603},
    "Nepal": {"iso_alpha": "NPL", "lat": 28.3949, "lon": 84.1240, "population": 30547580},
    "Venezuela": {"iso_alpha": "VEN", "lat": 6.4238, "lon": -66.5897, "population": 28436426},
    "Madagascar": {"iso_alpha": "MDG", "lat": -18.7669, "lon": 46.8691, "population": 28915653},
    "Cameroon": {"iso_alpha": "CMR", "lat": 3.8480, "lon": 11.5021, "population": 28249251},
    "Ivory Coast": {"iso_alpha": "CIV", "lat": 7.5400, "lon": -5.5471, "population": 27654529},
    "North Korea": {"iso_alpha": "PRK", "lat": 40.3399, "lon": 127.5101, "population": 26160821},
    "Senegal": {"iso_alpha": "SEN", "lat": 14.4974, "lon": -14.4524, "population": 17316449},
    "Somalia": {"iso_alpha": "SOM", "lat": 5.1521, "lon": 46.1996, "population": 18143378},
    "Zimbabwe": {"iso_alpha": "ZWE", "lat": -19.0154, "lon": 29.1549, "population": 16665409},
    "Sri Lanka": {"iso_alpha": "LKA", "lat": 7.8731, "lon": 80.7718, "population": 21497310},
    "Mali": {"iso_alpha": "MLI", "lat": 17.5707, "lon": -3.9962, "population": 22405178},
    "Burkina Faso": {"iso_alpha": "BFA", "lat": 12.2383, "lon": -1.5616, "population": 22673762},
    "Syria": {"iso_alpha": "SYR", "lat": 34.8021, "lon": 38.9968, "population": 18269868},
    "Malawi": {"iso_alpha": "MWI", "lat": -13.2543, "lon": 34.3015, "population": 20195619},
    "Chile": {"iso_alpha": "CHL", "lat": -35.6751, "lon": -71.5430, "population": 19838061},
    "Australia": {"iso_alpha": "AUS", "lat": -25.2744, "lon": 133.7751, "population": 26465917},
    "Kazakhstan": {"iso_alpha": "KAZ", "lat": 48.0196, "lon": 66.9237, "population": 20406862},
    "Ecuador": {"iso_alpha": "ECU", "lat": -1.8312, "lon": -78.1834, "population": 18190484},
    "Zambia": {"iso_alpha": "ZMB", "lat": -13.1339, "lon": 27.8493, "population": 20216417},
    "Senegal": {"iso_alpha": "SEN", "lat": 14.4974, "lon": -14.4524, "population": 17316449},
    "Guatemala": {"iso_alpha": "GTM", "lat": 15.7835, "lon": -90.2308, "population": 18054816},
    "Guinea": {"iso_alpha": "GIN", "lat": 9.9456, "lon": -9.6966, "population": 13665541},
    "South Korea": {"iso_alpha": "KOR", "lat": 35.9078, "lon": 127.7669, "population": 51780579},
    "Spain": {"iso_alpha": "ESP", "lat": 40.4637, "lon": -3.7492, "population": 47615034},
    "Belgium": {"iso_alpha": "BEL", "lat": 50.5039, "lon": 4.4699, "population": 11590324},
    "Greece": {"iso_alpha": "GRC", "lat": 39.0742, "lon": 21.8243, "population": 10533984},
    "Portugal": {"iso_alpha": "PRT", "lat": 39.3999, "lon": -8.2245, "population": 10532564},
    "Sweden": {"iso_alpha": "SWE", "lat": 60.1282, "lon": 18.6435, "population": 10593151},
    "Hong Kong": {"iso_alpha": "HKG", "lat": 22.3193, "lon": 114.1694, "population": 7512000},
    "Austria": {"iso_alpha": "AUT", "lat": 47.5162, "lon": 14.5501, "population": 9121023},
    "Israel": {"iso_alpha": "ISR", "lat": 31.0461, "lon": 34.8516, "population": 9038309},
    "Serbia": {"iso_alpha": "SRB", "lat": 44.0165, "lon": 21.0059, "population": 6552000},
    "Bulgaria": {"iso_alpha": "BGR", "lat": 42.7339, "lon": 25.4858, "population": 6839000},
    "Cuba": {"iso_alpha": "CUB", "lat": 21.5218, "lon": -77.7812, "population": 10987317},
    "Dominican Republic": {"iso_alpha": "DOM", "lat": 18.7357, "lon": -70.1627, "population": 11056884},
    "Czech Republic": {"iso_alpha": "CZE", "lat": 49.8175, "lon": 15.4730, "population": 10510785},
    "Haiti": {"iso_alpha": "HTI", "lat": 18.9712, "lon": -72.2852, "population": 11263077},
    "New Zealand": {"iso_alpha": "NZL", "lat": -40.9006, "lon": 174.8860, "population": 5188000},
    "Norway": {"iso_alpha": "NOR", "lat": 60.4720, "lon": 8.4689, "population": 5490374},
    "Costa Rica": {"iso_alpha": "CRI", "lat": 9.7489, "lon": -83.7534, "population": 5334052},
    "Ireland": {"iso_alpha": "IRL", "lat": 53.4129, "lon": -8.2439, "population": 5263277},
    "Singapore": {"iso_alpha": "SGP", "lat": 1.3521, "lon": 103.8198, "population": 5916100},
}


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


def get_mock_data(spark=None, days=365, regions=None):
    """
    Generate synthetic COVID-19-like dataset with real country data.
    
    Returns (spark_df, pandas_df) tuple. If spark is unavailable, spark_df is None.
    Columns: Date, Country, ISO_Alpha, Latitude, Longitude, Confirmed, Deaths, Recoveries, Active_Per_100k
    """
    if regions is None:
        regions = len(COUNTRIES_DATA)
    
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days - 1)

    # Use real country data
    country_list = list(COUNTRIES_DATA.items())[:regions]
    
    rows = []
    for country_name, country_info in country_list:
        iso_alpha = country_info["iso_alpha"]
        lat = country_info["lat"]
        lon = country_info["lon"]
        pop = country_info["population"]
        
        baseline_cases = random.randint(5000, 500_000)
        baseline_deaths = int(baseline_cases * random.uniform(0.005, 0.03))
        baseline_recoveries = int(baseline_cases * random.uniform(0.6, 0.95))

        for day_idx in range(days):
            date = start_date + timedelta(days=day_idx)

            # simple epidemic curve noise + seasonality
            seasonal = 1 + 0.25 * (1 + math.sin(day_idx / 30.0))
            growth = 1 + (day_idx / (days * 10.0))
            confirmed = max(0, int(baseline_cases * seasonal * growth * random.uniform(0.8, 1.2)))
            deaths = max(0, int(confirmed * random.uniform(0.001, 0.03)))
            recoveries = max(0, int(confirmed * random.uniform(0.5, 0.95)))

            active_per_100k = (confirmed - recoveries) / max(1, (pop / 100_000))

            rows.append((date, country_name, iso_alpha, float(lat), float(lon), int(confirmed), int(deaths), int(recoveries), float(active_per_100k)))

    pdf = pd.DataFrame(rows, columns=["Date", "Country", "ISO_Alpha", "Latitude", "Longitude", "Confirmed", "Deaths", "Recoveries", "Active_Per_100k"]) 

    # Try to convert to Spark if available
    sdf = None
    if spark is not None:
        try:
            schema = StructType([
                StructField("Date", DateType(), False),
                StructField("Country", StringType(), False),
                StructField("ISO_Alpha", StringType(), False),
                StructField("Latitude", DoubleType(), False),
                StructField("Longitude", DoubleType(), False),
                StructField("Confirmed", IntegerType(), False),
                StructField("Deaths", IntegerType(), False),
                StructField("Recoveries", IntegerType(), False),
                StructField("Active_Per_100k", DoubleType(), False),
            ])
            sdf = spark.createDataFrame(pdf, schema=schema)
            sdf = sdf.repartition(4)
            sdf.cache()
            sdf.count()  # materialize cache for snappy response in the app
        except Exception:
            sdf = None
    
    return sdf, pdf


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
    """Aggregate by country to get recent snapshots for mapping and tables."""
    # Use latest date per country
    window_max = sdf.groupBy("Country").agg(F.max("Date").alias("latest_date"))
    joined = sdf.join(window_max, (sdf.Country == window_max.Country) & (sdf.Date == window_max.latest_date), how="inner").select(sdf["*"])

    countries = (
        joined.groupBy("Country", "ISO_Alpha", "Latitude", "Longitude")
        .agg(
            F.sum("Confirmed").alias("confirmed"),
            F.sum("Deaths").alias("deaths"),
            F.sum("Recoveries").alias("recoveries"),
            F.avg("Active_Per_100k").alias("active_per_100k"),
        )
        .orderBy(F.desc("confirmed"))
    )

    return countries.toPandas()


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
    # take latest record per country
    latest = pdf.sort_values(["Country", "Date"]).groupby("Country").tail(1)

    total_cases = int(latest["Confirmed"].sum())
    total_deaths = int(latest["Deaths"].sum())
    total_recoveries = int(latest["Recoveries"].sum()) if "Recoveries" in latest.columns else 0
    total_active = int(latest.get("Active", (latest["Confirmed"] - latest.get("Recoveries", 0))).sum())

    recovery_rate = float(total_recoveries) / total_cases if total_cases else 0.0
    active_rate = float(total_active) / total_cases if total_cases else 0.0

    return {
        "total_confirmed": total_cases,
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
    """Produce latest snapshot aggregated by country from pandas DataFrame for mapping and tables."""
    latest = pdf.sort_values(["Country", "Date"]).groupby("Country").tail(1)
    out = latest[["Country", "ISO_Alpha", "Latitude", "Longitude", "Confirmed", "Recoveries", "Active_Per_100k"]].copy()
    out["confirmed"] = out["Confirmed"]
    out["active_per_100k"] = out["Active_Per_100k"]
    return out[["Country", "ISO_Alpha", "Latitude", "Longitude", "confirmed", "active_per_100k"]]
