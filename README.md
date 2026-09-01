# GlacioGuard

**GlacioGuard** is a scalable, multi-modal early-warning platform designed for real-time monitoring of glacial lakes and assessing Glacial Lake Outburst Flood (GLOF) risk. It integrates satellite imagery, high-frequency weather data, snow cover metrics, and terrain topography to construct a robust, deterministic observation engine.

---

## 🏔️ System Architecture

The platform is designed in distinct, decoupled stages.

### Step 1 — Master Lake Inventory
Builds a clean, canonical inventory of Indian glacial lakes from the best available scientific datasets. 
- **Data Source:** PANGAEA (Kumar & Vijay, 2026) - *Inventory of Glacial Lakes in High Mountain Asia*
- **Filtering:** Filters 31,698 lakes to strictly those within Indian boundaries using Natural Earth geometry.
- **Output:** Assigns a deterministic, stable identifier (`lake_uid = GLACIOGUARD_IN_XXXXXX`) used downstream.

### Step 2 — Historical + Near-Real-Time Observation Engine
Establishes a scalable ingestion and processing pipeline connecting each `lake_uid` to critical environmental observations. The pipeline handles data extraction, feature engineering, and temporal alignment across disparate datasets with robust quality controls.

#### Supported Data Modalities:
- **Terrain:** Static SRTM 30m DEM elevation, slope, and aspect via OpenTopography API.
- **Weather:** Hourly ERA5-Land (temperature, precipitation, snowfall) via Copernicus CDS API (NetCDF/ZIP).
- **Satellite:** Sentinel-2 L2A via CDSE STAC (water area tracking & cloud masking).
- **Snow:** MODIS Snow Cover (MOD10A1) via NASA Earthdata.
- **Precipitation:** High-frequency GPM IMERG via NASA Earthdata.

---

## ⚙️ Setup and Installation

### 1. Environment Setup
GlacioGuard is built on Python and heavily utilizes `pandas`, `geopandas`, and `xarray`.

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
# OR Activate (Unix)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Authentication & Credentials
The Observation Engine aggregates data from multiple secure agencies. You can run the pipeline without all keys (unavailable services will gracefully skip), but for full functionality, configure the following credentials in your environment or via a `.env` file:

- **Sentinel-2 (Copernicus Data Space):** Set `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`.
- **ERA5-Land (Copernicus CDS):** Set `CDSAPI_URL` and `CDSAPI_KEY` (or configure `~/.cdsapirc`).
- **NASA Earthdata (MODIS & GPM):** Set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` (or configure `~/.netrc`).
- **OpenTopography:** Set `OPENTOPOGRAPHY_API_KEY`.

---

## 🚀 Usage

The project utilizes modular scripts for each step, and a unified runner for Step 2.

### Step 1: Initialize Inventory
```bash
python ingestion/pangaea_lake_inventory.py
```
*(The first run downloads ~300 MB of source data. Subsequent runs use cache unless `--force` is passed.)*

### Step 2: Observation Engine
You can run Step 2 for specific targets or the entire inventory.

```bash
# Validate credentials and API access gracefully
python run_step2.py --check-access

# Run a test pass for a deterministically auto-selected subset of lakes
python run_step2.py --test

# Run the pipeline for a specific lake
python run_step2.py --lake-id GLACIOGUARD_IN_001593

# Run the full pipeline for all lakes
python run_step2.py --all
```

---

## 📁 Project Structure & Outputs

```
GlacioGuard/
├── config/                      # YAML configuration (dates, bounds, sources)
├── ingestion/                   # Raw data downloaders and API clients
│   ├── pangaea_lake_inventory.py
│   ├── era5_land.py
│   ├── gpm_imerg.py
│   ├── modis_snow.py
│   └── sentinel2_observations.py
├── processing/                  # Feature engineering and temporal alignment
│   ├── temporal_alignment.py
│   ├── feature_engineering.py
│   └── quality_control.py
├── tests/                       # Automated leakage and temporal checks
└── run_step2.py                 # Unified Stage 2 pipeline runner
```

### Data Pipeline Outputs (`data/processed/`)
All output files are deterministic, safely updated atomically, and deduplicated via `observation_id`.
- `india_glacial_lakes_2022.geojson` / `.csv` (Canonical Inventory)
- `lake_terrain.parquet` (Static SRTM elevation profiles)
- `lake_weather_observations.parquet` (ERA5-Land hourly temps/precip)
- `lake_satellite_observations.parquet` (Sentinel-2 metrics)
- `lake_snow_observations.parquet` (MODIS NDSI tracking)
- `lake_precipitation_highfreq.parquet` (GPM IMERG records)
- `master_temporal_grid.parquet` (The final dynamically-aligned master feature table, guaranteed 0-temporal-leakage)

---

## 🔮 Next Steps (Step 3)

**Step 3** is currently pending. It will utilize the unified `master_temporal_grid.parquet` to:
- Handle historical hazard event tagging.
- Run baseline modeling and anomaly detection.
- Train predictive Machine Learning algorithms.
- Establish active GLOF risk hazard scoring & alerting mechanisms.
