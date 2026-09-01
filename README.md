# 🏔️ GlacioGuard

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**GlacioGuard** is a highly scalable, multi-modal early-warning platform designed for real-time monitoring of glacial lakes and assessing **Glacial Lake Outburst Flood (GLOF)** risk in High Mountain Asia (HMA) and the Indian Himalayas.

By integrating multi-spectral satellite imagery, high-frequency weather data, snow cover metrics, and high-resolution terrain topography, GlacioGuard constructs a robust, deterministic observation engine to fuel hazard baseline modeling and anomaly detection.

---

## 📑 Table of Contents
- [System Architecture](#-system-architecture)
  - [Step 1: Master Lake Inventory](#step-1--master-lake-inventory)
  - [Step 2: Observation Engine](#step-2--historical--near-real-time-observation-engine)
- [Methodology & Data integrity](#-methodology--data-integrity)
- [Setup and Installation](#-setup-and-installation)
- [Configuration & Credentials](#-configuration--credentials)
- [Usage Guide](#-usage-guide)
- [Data Schemas & Outputs](#-data-schemas--outputs)
- [Next Steps (Step 3)](#-next-steps)

---

## 🏗️ System Architecture

The platform is designed in distinct, decoupled stages to ensure deterministic reproducibility and strict data provenance.

### Step 1 — Master Lake Inventory
Builds a clean, canonical inventory of Indian glacial lakes from the best available scientific datasets. 
- **Data Source:** PANGAEA (Kumar & Vijay, 2026) - *Inventory of Glacial Lakes in High Mountain Asia*
- **Filtering Pipeline:** Geographically filters 31,698 lakes to strictly those within Indian boundaries using Natural Earth geometry (`Admin-0`/`Admin-1`).
- **Identity Assignment:** Assigns a deterministic, stable identifier (`lake_uid = GLACIOGUARD_IN_XXXXXX`) used universally across all downstream pipelines.
- **Scale:** Processes ~300 MB of source data into lightweight `.parquet` and `.geojson` canonical stores.

### Step 2 — Historical + Near-Real-Time Observation Engine
Establishes a massively scalable ingestion and processing pipeline connecting each `lake_uid` to critical, time-variant environmental observations.

#### Supported Data Modalities:
1. **Terrain (Topography):** Static SRTM 30m DEM elevation, slope, and aspect profiles via **OpenTopography API**.
2. **Weather (Meteorology):** Hourly temperature, precipitation, and snowfall metrics via **Copernicus CDS API** (ERA5-Land).
3. **Satellite (Optical):** Water area tracking and multi-spectral indices via **Copernicus Data Space (CDSE) STAC** (Sentinel-2 L2A).
4. **Snow (Cryosphere):** Normalized Difference Snow Index (NDSI) tracking via **NASA Earthdata** (MODIS MOD10A1).
5. **Precipitation (High-Frequency):** Sub-daily extreme rain events via **NASA Earthdata** (GPM IMERG).

---

## 🛡️ Methodology & Data Integrity

GlacioGuard places an extreme emphasis on data integrity for critical early-warning contexts:
- **0-Temporal Leakage Guarantee:** The Temporal Alignment engine (`processing/temporal_alignment.py`) utilizes strictly backward `merge_asof` joins. Future data is *never* allowed to leak into past observation features, ensuring safe ML training downstream.
- **Atomic Writes:** All data operations write to temporary `.tmp.parquet` buffers, only replacing the canonical data store if validation succeeds.
- **Deduplication:** Every row is assigned a deterministic SHA-256 `observation_id` (hashed from the `lake_uid`, `timestamp`, and `source`), ensuring reruns do not duplicate or corrupt rows.
- **Cloud Masking:** Satellite extraction automatically skips or masks scenes exceeding strict cloud cover thresholds defined in configuration.

---

## ⚙️ Setup and Installation

### 1. Environment Setup
GlacioGuard is built heavily on the `geopandas`, `xarray`, and `pandas` stack. An isolated virtual environment is strongly recommended.

```bash
# Clone the repository
git clone https://github.com/HimanshuIITP/GlacioGuard.git
cd GlacioGuard

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate
# Activate (Unix/macOS)
source venv/bin/activate

# Install strictly pinned reproducible dependencies
pip install -r requirements.txt
```

---

## 🔑 Configuration & Credentials

The Observation Engine aggregates data from multiple secure science agencies. You can run the pipeline without all keys (unavailable services will gracefully skip), but for full production functionality, configure the following credentials:

### Method A: Environment Variables (or `.env` file)
Create a `.env` file in the root directory:
```env
# Copernicus Data Space (Sentinel-2)
CDSE_CLIENT_ID=your_client_id
CDSE_CLIENT_SECRET=your_client_secret

# Copernicus CDS (ERA5-Land)
CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
CDSAPI_KEY=your_cds_api_key

# NASA Earthdata (MODIS & GPM)
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password

# OpenTopography (SRTM DEM)
OPENTOPOGRAPHY_API_KEY=your_api_key
```

### Method B: Native Config Files
Alternatively, standard configuration files in your home directory are respected:
- `~/.cdsapirc` (for ERA5-Land)
- `~/.netrc` (for NASA Earthdata)

### Pipeline Tuning
Modify parameters inside the `config/` directory:
- **`observation_config.yaml`**: Date ranges, strict cloud cover thresholds, and spatial buffer sizes.
- **`test_lakes.yaml`**: Pre-defined UIDs for rapid pipeline validation.

---

## 🚀 Usage Guide

The project utilizes modular scripts for each step, alongside a unified runner for Step 2.

### Step 1: Initialize the Lake Inventory
```bash
python ingestion/pangaea_lake_inventory.py
```
*(Data is cached in `data/raw/` to speed up future runs. Use `--force` to re-download).*

### Step 2: Run the Observation Engine
The `run_step2.py` orchestrator executes extraction, feature engineering, and temporal alignment across all available data modalities.

```bash
# 1. Validate credentials and API access gracefully
python run_step2.py --check-access

# 2. Run a fast test pass (uses `config/test_lakes.yaml` for a small lake subset)
python run_step2.py --test

# 3. Process a specific glacial lake by UID
python run_step2.py --lake-id GLACIOGUARD_IN_001593

# 4. Execute the full production pipeline for all Indian lakes
python run_step2.py --all
```

---

## 📁 Data Schemas & Outputs

All data outputs are routed to `data/processed/`. The capstone artifact of Step 2 is the `master_temporal_grid.parquet`.

### Key Output Files:
- `india_glacial_lakes_2022.geojson` / `.csv`: The canonical spatial inventory.
- `lake_terrain.parquet`: Static SRTM elevation profiles.
- `lake_weather_observations.parquet`: Extracted hourly meteorology.
- `lake_satellite_observations.parquet`: Multi-spectral optical observations.
- `lake_snow_observations.parquet`: NDSI snow cover ratios.
- `lake_precipitation_highfreq.parquet`: Sub-daily precipitation events.
- **`master_temporal_grid.parquet`**: The finalized, dynamically-aligned master feature table.

### `master_temporal_grid.parquet` Schema Highlight:
| Field | Type | Description |
|-------|------|-------------|
| `lake_uid` | `string` | Canonical GlacioGuard identifier |
| `reference_timestamp` | `datetime64[ns, UTC]` | Hourly alignment bucket |
| `temperature_2m_c` | `float64` | Backwards-aligned temperature (°C) |
| `precipitation_mm` | `float64` | Backwards-aligned precipitation |
| `snow_fraction` | `float64` | Backwards-aligned NDSI fractional snow cover |
| `actual_*_observation`| `boolean` | Flags indicating if real data exists at this precise timestamp (to differentiate from back-filled sparse data). |

---

## 🔮 Next Steps

**Step 3 (Hazard Modeling & Alerting)** is currently pending implementation. It will utilize the unified `master_temporal_grid.parquet` to:
- Identify and tag historical hazard anomalies.
- Establish baseline environmental envelopes for individual glacial lakes.
- Train predictive Machine Learning algorithms.
- Establish real-time GLOF risk hazard scoring and dispatch alerting mechanisms.

---

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.
Source data (PANGAEA) operates under CC-BY-4.0.
