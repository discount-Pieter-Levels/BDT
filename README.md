# Large-Scale Analysis of COVID-19 Epidemiological Data

A Streamlit dashboard for analyzing synthetic COVID-19 epidemiological data using PySpark (with pandas fallback).

## Features

- **Palantir-style dark UI** with deep theme (`#11151c` background, `#1a202c` cards)
- **Interactive global map** showing case density by region
- **Dual-axis charts** displaying daily vs. cumulative cases
- **Regional monthly breakdown** with percentage growth metrics
- **PySpark-backed** heavy lifting on local[*] with automatic pandas fallback if Spark unavailable
- **Synthetic data generation** for immediate demo without external files

## Setup

### Option 1: Using pip (easiest)

```bash
pip3 install -r requirements.txt
streamlit run app.py
```

### Option 2: Virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # or on Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- Python 3.8+
- OpenJDK 11+ (for PySpark; optional—app falls back to pandas if Java unavailable)

### Checking Java

```bash
java -version
which java
```

If Java isn't installed on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk
```

## Architecture

### Files

- **`app.py`** — Main Streamlit dashboard with UI, CSS, charts, and tables
- **`data_processor.py`** — PySpark utilities for data generation and aggregation
- **`.streamlit/config.toml`** — Streamlit theme configuration
- **`requirements.txt`** — Python dependencies

### Data Flow

1. `init_spark()` tries to create a local PySpark session; returns `None` if Java unavailable
2. `get_mock_data(spark=...)` generates synthetic COVID data (always returns `(sdf, pdf)` tuple)
3. If Spark is available, uses PySpark for fast computation
4. If Spark fails, automatically falls back to pandas aggregations
5. All computations return consistent schemas; UI is identical either way

### Synthetic Data

Generated with realistic epidemic curves:
- **200 regions** (configurable) across the globe
- **180 days** of history (configurable)
- **Columns**: Date, Region, Latitude, Longitude, Confirmed, Deaths, Recoveries, Active_Per_100k, Retail_Sales_Impact

## UI Components

### Top Metric Bar
- **Overall Risk Level** — Color-coded (HIGH/MEDIUM/LOW) based on active rate
- **Total Cases** — Human-formatted with daily delta
- **Active Cases Per 100k** — Population-adjusted metric
- **Est. Recoveries** — Cumulative recoveries

### Main Layout
- **Left (60%)** — Interactive global map (Plotly Scattermapbox) colored by active cases per 100k
- **Right (40%)** — Tabbed interface:
  - **Tab 1: COVID Charts** — Dual-axis bar + line chart (daily vs. cumulative)
  - **Tab 2: Regional Data** — Monthly breakdown with growth % calculation

## Performance Notes

- Data generation: ~1–5s for 120 regions × 180 days
- PySpark queries: ~500ms (with caching)
- Pandas fallback: ~100–200ms
- Streamlit caching (10min TTL) prevents re-computation on reruns

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

```bash
pip3 install -r requirements.txt
```

### "JAVA_GATEWAY_EXITED" or Spark won't start

Java is unavailable. The app will automatically fall back to pandas. To enable PySpark:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Linux example
export PATH="$JAVA_HOME/bin:$PATH"
streamlit run app.py
```

### App loads but shows no data

Check that `data_processor.py` can be imported:

```bash
python3 -c "import data_processor; print('OK')"
```

If that works, refresh the Streamlit app in your browser.

## Customization

Edit these in `app.py` to adjust:

- **`load_data(days=180, regions=120)`** — Dataset size
- **CSS variables** — Colors, fonts, spacing
- **Chart options** — Zoom levels, color scales, templates

Edit in `data_processor.py`:

- **`_simulate_region_time_series()`** — Epidemic curve realism
- **`.config("spark.sql.shuffle.partitions", "4")`** — Parallelism

## License

Demo for "Large-Scale Analysis of COVID-19 Epidemiological Data" (Big Data Technologies course)
