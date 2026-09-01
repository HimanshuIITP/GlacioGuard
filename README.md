# GlacioGuard

Real-time multi-lake glacial hazard and GLOF early-warning platform.

## Current Status: Step 1 — Master Lake Inventory

The first stage builds a clean, canonical inventory of Indian glacial lakes
from the best available scientific dataset.

## Data Source

**PANGAEA — Kumar & Vijay (2026)**

- *Inventory of Glacial Lakes in High Mountain Asia for the Years 2016 and 2022*
- DOI: [10.1594/PANGAEA.983845](https://doi.org/10.1594/PANGAEA.983845)
- 31,698 glacial lakes across High Mountain Asia
- License: CC-BY-4.0
- Method: Landsat-8, Sentinel-1, Sentinel-2, Copernicus DEM

India boundary filtering uses
[Natural Earth](https://www.naturalearthdata.com/) 10m Admin-0/Admin-1
(public domain, de-facto boundaries).

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python ingestion/pangaea_lake_inventory.py
```

First run downloads ~300 MB of source data. Subsequent runs skip downloads
unless `--force` is passed.

## Outputs

```
data/processed/
├── india_glacial_lakes_2016.geojson   # 2016 epoch (median composites)
├── india_glacial_lakes_2016.csv
├── india_glacial_lakes_2022.geojson   # 2022 epoch (median composites)
├── india_glacial_lakes_2022.csv
└── inventory_quality_report.md
```

Each lake has a stable identifier (`lake_uid = GLACIOGUARD_IN_XXXXXX`) that
will be used by all future pipelines (satellite observations, weather,
seismic, predictions, alerts).

### Schema

| Field | Description |
|-------|-------------|
| `lake_uid` | Stable GlacioGuard identifier |
| `source_lake_id` | Original PANGAEA identifier |
| `observation_epoch` | 2016 or 2022 |
| `latitude` / `longitude` | Centroid (WGS84) |
| `geometry` | Lake polygon |
| `country` | India |
| `state_or_region` | From Natural Earth admin-1 |
| `elevation_m` | If available in source |
| `lake_area_km2` | Computed from geometry if not in source |
| `lake_type` | If available in source |
| `glacier_id` | If available in source |
| `river_basin` | If available in source |
| `boundary_status` | inside / boundary |
| `geometry_repair_status` | valid_original / repaired / unrepairable |
| `duplicate_status` | unique / exact_duplicate / near_duplicate_candidate |

## Assumptions and Limitations

- Natural Earth de-facto boundaries exclude Aksai Chin
- Lakes with ≥ 50% area inside India boundary classified as "inside"
- Boundary lakes (> 0% but < 50%) retained and flagged
- Cross-epoch UID matching uses 200 m centroid proximity
- Minimum detectable lake ≈ 20,000 m² (0.02 km²)
- See `inventory_quality_report.md` for detailed quality analysis

## Project Structure

```
GlacioGuard/
├── ingestion/
│   └── pangaea_lake_inventory.py    # Inventory ingestion pipeline
├── data/
│   ├── raw/                         # Downloaded source data (gitignored)
│   │   ├── pangaea/
│   │   │   ├── 2016/
│   │   │   └── 2022/
│   │   └── natural_earth/
│   └── processed/                   # Pipeline outputs
├── requirements.txt
└── README.md
```

## Current Status: Step 2 — Historical + Near-Real-Time Lake Observation Engine

Step 2 establishes a scalable ingestion and processing pipeline connecting each `lake_uid` to critical environmental observations. The pipeline handles:
- **Terrain**: Static SRTM 30m DEM elevation, slope, and aspect
- **Weather**: Hourly ERA5-Land (temperature, precipitation, snowfall)
- **Satellite**: Sentinel-2 L2A via CDSE STAC (water area tracking)
- **Snow**: MODIS Snow Cover (MOD10A1) via NASA Earthdata
- **Precipitation**: High-frequency GPM IMERG via NASA Earthdata

### Setup & Credentials
The pipeline supports `run_step2.py` as an end-to-end runner. It uses `--check-access` to validate environment credentials gracefully without failure if APIs are unavailable. 
To configure access:
- **Sentinel-2**: Set `CDSE_USERNAME` and `CDSE_PASSWORD`
- **ERA5-Land**: Create `~/.cdsapirc` or set `CDSAPI_URL` and `CDSAPI_KEY`
- **NASA Data**: Create `~/.netrc` or set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`
- **OpenTopography**: Set `OPENTOPOGRAPHY_API_KEY`

### Running Step 2

```bash
# Validate credentials
python run_step2.py --check-access

# Run for deterministically auto-selected test lakes
python run_step2.py --test

# Run for a specific lake
python run_step2.py --lake-id GLACIOGUARD_IN_001593

# Run full pipeline for all lakes
python run_step2.py --all
```

### Outputs
```
data/processed/
├── lake_terrain.parquet
├── lake_weather_observations.parquet
├── lake_satellite_observations.parquet
├── lake_snow_observations.parquet
├── lake_precipitation_highfreq.parquet
├── lake_baselines.parquet
├── lake_data_coverage.parquet
├── lake_observations.parquet
└── observation_quality_report.md
```

## Next Steps

**Step 3** will separately handle historical hazard events, feature selection, baseline/anomaly modeling, and GLOF risk prediction using the `lake_observations.parquet` master feature table.
