#!/usr/bin/env python3
"""
GlacioGuard — PANGAEA Glacial Lake Inventory Ingestion Pipeline
================================================================
Downloads the Kumar & Vijay (2026) High Mountain Asia glacial lake inventory
from PANGAEA, filters to India using Natural Earth boundaries, validates
geometries, detects duplicates, assigns stable lake UIDs, and produces
canonical GeoJSON + CSV outputs for both 2016 and 2022 epochs.

Source: https://doi.org/10.1594/PANGAEA.983845
License: CC-BY-4.0
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

PANGAEA_DOI = "10.1594/PANGAEA.983845"
PANGAEA_BASE_URL = "https://download.pangaea.de/dataset/983845/files"

# Two epochs, both using median-composite boundaries for consistency.
# Median composites minimise seasonal variation in lake extents.
EPOCHS = {
    "2016": {
        "prefix": "Glacial_lakes_2016_median",
        "extensions": [".shp", ".shx", ".dbf", ".prj", ".cpg"],
        "description": (
            "Median composite boundaries from 2016 Sentinel-2 imagery"
        ),
    },
    "2022": {
        "prefix": "Glacial_lakes_2022_median",
        "extensions": [".shp", ".shx", ".dbf", ".prj", ".cpg"],
        "description": (
            "Median composite boundaries from 2022 Sentinel-2 imagery"
        ),
    },
}

NE_ADMIN0_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_0_countries.zip"
)
NE_ADMIN1_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_1_states_provinces.zip"
)

# UTM 44N — covers ~78-84 °E; adequate for northern-India lake centroids.
# Minor distortion at extreme east/west is acceptable for area-ratio and
# distance calculations on small polygons.
METRIC_CRS = "EPSG:32644"
WGS84 = "EPSG:4326"

# Approximate India bounding-box for coordinate sanity checks only.
INDIA_LAT_RANGE = (6.0, 37.5)
INDIA_LON_RANGE = (68.0, 98.0)

# A lake with >= 50 % of its area inside the India polygon is "inside".
BOUNDARY_INSIDE_THRESHOLD = 0.50

# Centroid distance thresholds (metres)
NEAR_DUPLICATE_THRESHOLD_M = 50
CROSS_EPOCH_MATCH_THRESHOLD_M = 200

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Download *url* to *dest*; skip if already present unless *force*."""
    if dest.exists() and not force:
        print(f"  [skip] {dest.name} (exists)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET  {url}")

    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    done = 0

    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=131_072):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(
                    f"\r    {done / 1_048_576:.1f} / "
                    f"{total / 1_048_576:.1f} MB "
                    f"({done * 100 // total}%)",
                    end="",
                    flush=True,
                )
    if total:
        print()
    print(f"  OK   {dest.name}  ({dest.stat().st_size / 1_048_576:.1f} MB)")
    return dest


def download_pangaea_epoch(epoch: str, force: bool = False) -> Path:
    """Download all shapefile components for one PANGAEA epoch."""
    cfg = EPOCHS[epoch]
    dest_dir = DATA_RAW / "pangaea" / epoch
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg["prefix"]
    for ext in cfg["extensions"]:
        fname = f"{prefix}{ext}"
        download_file(f"{PANGAEA_BASE_URL}/{fname}", dest_dir / fname, force)
    return dest_dir / f"{prefix}.shp"


def download_natural_earth(force: bool = False):
    """Download Natural Earth admin-0 and admin-1 ZIPs."""
    ne_dir = DATA_RAW / "natural_earth"
    ne_dir.mkdir(parents=True, exist_ok=True)
    a0 = download_file(NE_ADMIN0_URL, ne_dir / "ne_10m_admin_0_countries.zip", force)
    a1 = download_file(NE_ADMIN1_URL, ne_dir / "ne_10m_admin_1_states_provinces.zip", force)
    return a0, a1

# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------

def inspect_shapefile(path: Path, label: str) -> gpd.GeoDataFrame:
    """Load a shapefile and print diagnostic metadata."""
    print(f"\n--- {label} ---")
    print(f"  File : {path}")
    gdf = gpd.read_file(path)
    print(f"  CRS  : {gdf.crs}")
    print(f"  Rows : {len(gdf)}")
    gtypes = gdf.geometry.geom_type.unique().tolist()
    print(f"  Geom : {gtypes}")
    print(f"  Bounds: {gdf.total_bounds}")
    print(f"  Cols : {list(gdf.columns)}")
    for col in gdf.columns:
        if col == "geometry":
            continue
        nn = gdf[col].notna().sum()
        print(f"    {col:20s}  {str(gdf[col].dtype):10s}  "
              f"{nn}/{len(gdf)} non-null")
    return gdf

# ---------------------------------------------------------------------------
# India boundary
# ---------------------------------------------------------------------------

def load_india_boundary(admin0_zip: Path, admin1_zip: Path):
    """Return (india_admin0_gdf, india_states_gdf)."""
    print("\n--- Loading India boundary ---")

    a0 = gpd.read_file(f"zip://{admin0_zip}")
    india = a0[a0["ADMIN"] == "India"] if "ADMIN" in a0.columns else pd.DataFrame()
    if india.empty:
        india = a0[a0["NAME"] == "India"]
    if india.empty:
        raise RuntimeError("India not found in Natural Earth admin-0")
    print(f"  Admin-0 India polygon loaded")

    a1 = gpd.read_file(f"zip://{admin1_zip}")
    # Filter India states
    for col in ["admin", "ADMIN", "adm0_a3", "sov_a3"]:
        if col in a1.columns:
            vals = ["India"] if col in ("admin", "ADMIN") else ["IND"]
            states = a1[a1[col].isin(vals)]
            if not states.empty:
                break
    else:
        states = gpd.GeoDataFrame()

    print(f"  Admin-1 India states: {len(states)}")
    return india.copy(), states.copy()

# ---------------------------------------------------------------------------
# Geographic filtering
# ---------------------------------------------------------------------------

def classify_boundary_status(lakes, india, metric_crs):
    """Add boundary_status (inside / boundary / outside) to *lakes*."""
    print("\n--- Boundary classification ---")

    lakes_m = lakes.to_crs(metric_crs)
    india_m = india.to_crs(metric_crs)
    india_union = india_m.union_all()

    lake_area = lakes_m.geometry.area
    ix_area = lakes_m.geometry.intersection(india_union).area

    ratio = pd.Series(0.0, index=lakes.index)
    nz = lake_area > 0
    ratio[nz] = ix_area[nz] / lake_area[nz]

    status = pd.Series("outside", index=lakes.index, dtype="object")
    status[ratio >= BOUNDARY_INSIDE_THRESHOLD] = "inside"
    status[(ratio > 0) & (ratio < BOUNDARY_INSIDE_THRESHOLD)] = "boundary"

    out = lakes.copy()
    out["boundary_status"] = status
    out["india_overlap_ratio"] = ratio.round(4)

    vc = status.value_counts()
    for k in ("inside", "boundary", "outside"):
        print(f"  {k:10s}: {vc.get(k, 0)}")
    return out


def assign_states(lakes, india_states):
    """Assign state_or_region via centroid spatial join."""
    if india_states.empty:
        lakes = lakes.copy()
        lakes["state_or_region"] = pd.NA
        return lakes

    name_col = "name" if "name" in india_states.columns else "NAME"

    pts = lakes.to_crs(METRIC_CRS).copy()
    pts["geometry"] = pts.geometry.centroid
    pts = pts.to_crs(lakes.crs)

    joined = gpd.sjoin(
        pts[["geometry"]],
        india_states[[name_col, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]

    lakes = lakes.copy()
    lakes["state_or_region"] = joined[name_col].reindex(lakes.index).values
    n = lakes["state_or_region"].notna().sum()
    print(f"  States assigned: {n}/{len(lakes)}")
    return lakes

# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------

def validate_geometries(gdf):
    """Validate / repair geometries; add geometry_repair_status column."""
    print("\n--- Geometry validation ---")
    gdf = gdf.copy()

    status = pd.Series("valid_original", index=gdf.index, dtype="object")
    area_delta = pd.Series(0.0, index=gdf.index)

    valid = gdf.geometry.is_valid
    invalid = ~valid & gdf.geometry.notna()
    print(f"  Valid   : {valid.sum()}")
    print(f"  Invalid : {invalid.sum()}")

    for idx in gdf.index[invalid]:
        geom = gdf.at[idx, "geometry"]
        orig_area = geom.area
        try:
            fixed = make_valid(geom)
            if fixed.is_valid and not fixed.is_empty:
                if orig_area > 0:
                    area_delta[idx] = abs(fixed.area - orig_area) / orig_area
                gdf.at[idx, "geometry"] = fixed
                status[idx] = "repaired"
            else:
                status[idx] = "unrepairable"
        except Exception:
            status[idx] = "unrepairable"

    gdf["geometry_repair_status"] = status
    gdf["repair_area_change"] = area_delta.round(6)

    vc = status.value_counts()
    print(f"  Repaired    : {vc.get('repaired', 0)}")
    print(f"  Unrepairable: {vc.get('unrepairable', 0)}")
    return gdf

# ---------------------------------------------------------------------------
# Coordinate validation
# ---------------------------------------------------------------------------

def validate_coordinates(gdf):
    """Flag centroids outside expected India bounding box."""
    print("\n--- Coordinate sanity check ---")
    gdf = gdf.copy()

    ok = (
        gdf["latitude"].between(*INDIA_LAT_RANGE)
        & gdf["longitude"].between(*INDIA_LON_RANGE)
    )
    gdf["coordinate_flag"] = np.where(ok, "normal", "outside_expected_range")
    n_bad = (~ok).sum()
    if n_bad:
        print(f"  WARNING: {n_bad} centroids outside expected range")
    else:
        print(f"  All {len(gdf)} centroids OK")
    return gdf

# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def detect_duplicates(gdf, metric_crs):
    """Detect exact and near-duplicate lakes."""
    print("\n--- Duplicate detection ---")
    gdf = gdf.copy()
    gdf["duplicate_status"] = "unique"

    # Exact: identical WKT
    wkt = gdf.geometry.apply(lambda g: g.wkt if g else None)
    exact = wkt.duplicated(keep="first")
    gdf.loc[exact, "duplicate_status"] = "exact_duplicate"
    print(f"  Exact duplicates: {exact.sum()}")

    # Near: centroid distance in metric CRS
    check = gdf["duplicate_status"] != "exact_duplicate"
    idxs = gdf.index[check].tolist()

    if len(idxs) < 2:
        print(f"  Near-duplicate candidates: 0")
        return gdf

    gdf_m = gdf.loc[idxs].to_crs(metric_crs)
    centroids = gdf_m.geometry.centroid

    from shapely import STRtree

    geoms = centroids.values
    tree = STRtree(geoms)
    near_set = set()

    for i, idx in enumerate(idxs):
        buf = geoms[i].buffer(NEAR_DUPLICATE_THRESHOLD_M)
        hits = tree.query(buf)
        for j in hits:
            if j <= i:
                continue
            if geoms[i].distance(geoms[j]) <= NEAR_DUPLICATE_THRESHOLD_M:
                near_set.add(idxs[j])

    for idx in near_set:
        if gdf.at[idx, "duplicate_status"] == "unique":
            gdf.at[idx, "duplicate_status"] = "near_duplicate_candidate"

    print(f"  Near-duplicate candidates: {len(near_set)}")
    return gdf

# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------

def _find_col(gdf, candidates):
    """Return first matching column name (case-insensitive) or None."""
    lower_map = {c.lower(): c for c in gdf.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize_schema(gdf, epoch, access_date):
    """Map source columns to canonical GlacioGuard schema."""
    print(f"\n--- Schema normalisation ({epoch}) ---")
    gdf = gdf.copy()

    # Ensure WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84)

    # Centroid coordinates — project to metric CRS to avoid geographic centroid warning
    centroids_m = gdf.to_crs(METRIC_CRS).geometry.centroid
    centroids_wgs = centroids_m.to_crs(WGS84)
    gdf["latitude"] = centroids_wgs.y.round(6)
    gdf["longitude"] = centroids_wgs.x.round(6)

    # -- Source lake ID --
    id_col = _find_col(gdf, [
        "GLIMS_ID", "glims_id", "Lake_ID", "lake_id",
        "ID", "FID", "OBJECTID", "GL_ID",
    ])
    if id_col:
        gdf["source_lake_id"] = gdf[id_col].astype(str)
        print(f"  source_lake_id <- {id_col}")
    else:
        gdf["source_lake_id"] = [f"ROW_{i}" for i in range(len(gdf))]
        print(f"  source_lake_id <- row index (no ID column found)")

    # -- Elevation --
    el_col = _find_col(gdf, [
        "Lake_Elev", "lake_elev", "elevation", "Elevation",
        "elev", "Elev", "DEM", "dem", "altitude", "Alt",
        "Z_mean", "Z_med",
    ])
    if el_col:
        gdf["elevation_m"] = pd.to_numeric(gdf[el_col], errors="coerce")
        print(f"  elevation_m <- {el_col}")
    else:
        gdf["elevation_m"] = pd.NA
        print(f"  elevation_m <- N/A")

    # -- Area --
    area_col = _find_col(gdf, [
        "Area_km2", "area_km2", "Area", "area",
        "Lake_Area", "lake_area", "AREA_SQKM",
    ])
    if area_col:
        raw = pd.to_numeric(gdf[area_col], errors="coerce")
        # Heuristic: if median > 100 assume m², convert to km²
        if raw.median() > 100:
            gdf["lake_area_km2"] = (raw / 1e6).round(6)
            print(f"  lake_area_km2 <- {area_col} (converted m2 -> km2)")
        else:
            gdf["lake_area_km2"] = raw.round(6)
            print(f"  lake_area_km2 <- {area_col}")
    else:
        # Compute from geometry
        gdf_m = gdf.to_crs(METRIC_CRS)
        gdf["lake_area_km2"] = (gdf_m.geometry.area / 1e6).round(6)
        print(f"  lake_area_km2 <- computed from geometry")

    # -- Lake type --
    lt_col = _find_col(gdf, ["lake_type", "Lake_Type", "type", "GL_Type"])
    gdf["lake_type"] = gdf[lt_col] if lt_col else pd.NA
    if lt_col:
        print(f"  lake_type <- {lt_col}")

    # -- Glacier ID --
    gl_col = _find_col(gdf, ["glacier_id", "Glacier_ID", "RGI_ID", "rgi_id"])
    gdf["glacier_id"] = gdf[gl_col] if gl_col else pd.NA
    if gl_col:
        print(f"  glacier_id <- {gl_col}")

    # -- River basin --
    rb_col = _find_col(gdf, ["river_basin", "River_Basin", "basin", "Basin", "Sub_Basin"])
    gdf["river_basin"] = gdf[rb_col] if rb_col else pd.NA
    if rb_col:
        print(f"  river_basin <- {rb_col}")

    # -- Fixed metadata --
    gdf["observation_epoch"] = epoch
    gdf["country"] = "India"
    gdf["source_dataset"] = "PANGAEA Kumar & Vijay (2026)"
    gdf["source_version"] = "2026-01-23"
    gdf["source_url"] = f"https://doi.org/{PANGAEA_DOI}"
    gdf["license"] = "CC-BY-4.0"
    gdf["access_date"] = access_date

    print(f"  {len(gdf)} records normalised")
    return gdf

# ---------------------------------------------------------------------------
# Lake UID assignment
# ---------------------------------------------------------------------------

def assign_lake_uids(gdf_2022, gdf_2016, metric_crs):
    """
    Assign stable GLACIOGUARD_IN_XXXXXX UIDs.
    2022 is the primary epoch; 2016 lakes matched by centroid proximity.
    """
    print("\n--- Lake UID assignment ---")
    match_stats = {"matched_2016": 0, "unmatched_2016": 0}

    gdf_2022 = gdf_2022.copy()
    gdf_2022["lake_uid"] = [
        f"GLACIOGUARD_IN_{i + 1:06d}" for i in range(len(gdf_2022))
    ]
    print(f"  2022: {len(gdf_2022)} UIDs assigned")

    next_id = len(gdf_2022) + 1

    if gdf_2016 is None or gdf_2016.empty:
        return gdf_2022, gdf_2016, match_stats

    gdf_2016 = gdf_2016.copy()

    c22 = gdf_2022.to_crs(metric_crs).geometry.centroid
    c16 = gdf_2016.to_crs(metric_crs).geometry.centroid

    from shapely import STRtree

    tree = STRtree(c22.values)
    uids = []

    for i in range(len(gdf_2016)):
        pt = c16.iloc[i]
        nearest = tree.nearest(pt)
        dist = pt.distance(c22.iloc[nearest])
        if dist <= CROSS_EPOCH_MATCH_THRESHOLD_M:
            uids.append(gdf_2022.iloc[nearest]["lake_uid"])
            match_stats["matched_2016"] += 1
        else:
            uids.append(f"GLACIOGUARD_IN_{next_id:06d}")
            next_id += 1
            match_stats["unmatched_2016"] += 1

    gdf_2016["lake_uid"] = uids
    print(
        f"  2016: {match_stats['matched_2016']} matched, "
        f"{match_stats['unmatched_2016']} new UIDs"
    )
    return gdf_2022, gdf_2016, match_stats

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

CANONICAL_COLS = [
    "lake_uid", "source_lake_id", "source_dataset", "source_version",
    "observation_epoch", "country", "state_or_region",
    "latitude", "longitude", "geometry",
    "elevation_m", "lake_area_km2", "lake_type", "glacier_id", "river_basin",
    "geometry_repair_status", "duplicate_status", "boundary_status",
    "india_overlap_ratio", "coordinate_flag",
    "source_url", "license", "access_date",
]


def write_outputs(gdf, epoch):
    """Write GeoJSON and CSV for one epoch."""
    print(f"\n--- Writing {epoch} outputs ---")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Column ordering
    cols = [c for c in CANONICAL_COLS if c in gdf.columns]
    extra = [c for c in gdf.columns if c not in cols]
    out = gdf[cols + extra].copy()

    if out.crs and out.crs.to_epsg() != 4326:
        out = out.to_crs(WGS84)

    # Keep inside + boundary only
    out = out[out["boundary_status"].isin(["inside", "boundary"])].copy()

    # GeoJSON — exclude unrepairable
    gj = out[out["geometry_repair_status"] != "unrepairable"].copy()
    gj_path = DATA_PROCESSED / f"india_glacial_lakes_{epoch}.geojson"
    gj.to_file(gj_path, driver="GeoJSON")
    print(f"  GeoJSON : {gj_path.name}  ({len(gj)} features)")

    # CSV -- geometry as WKT
    csv_df = pd.DataFrame(out.drop(columns="geometry"))
    csv_df["geometry"] = out.geometry.apply(
        lambda g: g.wkt if g is not None else None
    )
    csv_path = DATA_PROCESSED / f"india_glacial_lakes_{epoch}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  CSV     : {csv_path.name}  ({len(csv_df)} rows)")

    return len(gj), len(csv_df)

# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def generate_quality_report(stats, path):
    """Write inventory_quality_report.md."""
    print("\n--- Quality report ---")
    lines = [
        "# GlacioGuard — Glacial Lake Inventory Quality Report",
        "",
        f"Generated: {stats['access_date']}",
        "",
        "## Source Provenance",
        "",
        "### Primary Dataset",
        "",
        "| Field | Value |",
        "|-------|-------|",
        "| Dataset | Inventory of Glacial Lakes in High Mountain Asia "
        "for the Years 2016 and 2022 |",
        "| Authors | Kumar, Ravindra; Vijay, Saurabh |",
        f"| DOI | https://doi.org/{PANGAEA_DOI} |",
        "| Publisher | PANGAEA |",
        "| Published | 2026-01-23 |",
        "| License | CC-BY-4.0 |",
        f"| Access Date | {stats['access_date']} |",
        "| Method | Landsat-8, Sentinel-1, Sentinel-2, Copernicus DEM |",
        "",
        "### Boundary Dataset",
        "",
        "| Field | Value |",
        "|-------|-------|",
        "| Dataset | Natural Earth 10m Admin-0 + Admin-1 |",
        "| URL | https://www.naturalearthdata.com/ |",
        "| License | Public Domain |",
        "| Note | De-facto boundaries; does not resolve "
        "territorial disputes |",
        "",
    ]

    for epoch in ("2022", "2016"):
        es = stats.get(epoch, {})
        fc = es.get("final_count", 0)
        lines += [
            f"## Epoch: {epoch}",
            "",
            "### Source Counts",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Source records (HMA) | {es.get('source_records', '?')} |",
            f"| Inside India | {es.get('inside', '?')} |",
            f"| Boundary | {es.get('boundary', '?')} |",
            f"| Outside India | {es.get('outside', '?')} |",
            f"| India candidates | {es.get('india_candidates', '?')} |",
            "",
            "### Geometry Quality",
            "",
            "| Status | Count |",
            "|--------|-------|",
            f"| Valid original | {es.get('valid_original', '?')} |",
            f"| Repaired | {es.get('repaired', '?')} |",
            f"| Unrepairable | {es.get('unrepairable', '?')} |",
            "",
            "### Duplicate Analysis",
            "",
            "| Status | Count |",
            "|--------|-------|",
            f"| Unique | {es.get('unique', '?')} |",
            f"| Exact duplicates | {es.get('exact_duplicates', '?')} |",
            f"| Near-dup candidates | "
            f"{es.get('near_duplicate_candidates', '?')} |",
            "",
            "### Missing Data",
            "",
            "| Field | Null | Total |",
            "|-------|------|-------|",
        ]
        for fld in (
            "elevation_m", "lake_area_km2", "lake_type",
            "glacier_id", "river_basin", "state_or_region",
        ):
            nv = es.get(f"null_{fld}", "?")
            lines.append(f"| {fld} | {nv} | {fc} |")
        lines += [
            "",
            f"### Final inventory: **{fc}** Indian glacial lakes",
            "",
        ]

    ms = stats.get("match_stats", {})
    lines += [
        "## Cross-Epoch UID Matching",
        "",
        f"- Threshold: {CROSS_EPOCH_MATCH_THRESHOLD_M} m centroid distance",
        f"- 2016 matched to 2022: {ms.get('matched_2016', '?')}",
        f"- 2016 unmatched (new UID): {ms.get('unmatched_2016', '?')}",
        "",
        "## Known Limitations",
        "",
        "### Source",
        "- Dataset curated by PANGAEA but lacks final author approval",
        "- Minimum detectable lake ≈ 20 000 m² (0.02 km²)",
        "- Median composites smooth seasonal variation; "
        "transient lakes may be missed",
        "",
        "### Processing",
        "- Natural Earth de-facto boundaries; "
        "Aksai Chin lakes excluded",
        f"- Boundary threshold: {BOUNDARY_INSIDE_THRESHOLD * 100:.0f} % "
        "overlap",
        f"- Near-duplicate threshold: {NEAR_DUPLICATE_THRESHOLD_M} m; "
        "genuine close lakes may be flagged",
        f"- Cross-epoch match: {CROSS_EPOCH_MATCH_THRESHOLD_M} m; "
        "split/merged lakes may be mismatched",
        f"- Projection: {METRIC_CRS}; minor distortion at "
        "extreme longitudes",
        "",
        "### Assumptions",
        "- ≥ 50 % area inside Natural Earth India = 'inside'",
        "- Boundary lakes retained but flagged",
        "- lake_type, glacier_id, river_basin left null when "
        "absent in source",
        "- State assigned via Natural Earth admin-1 centroid join",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {path.name}")

# ---------------------------------------------------------------------------
# Epoch pipeline
# ---------------------------------------------------------------------------

def process_epoch(epoch, shp_path, india, india_states, access_date):
    """Process one epoch end-to-end; return (gdf, stats_dict)."""
    gdf = inspect_shapefile(shp_path, f"PANGAEA {epoch}")
    stats = {"source_records": len(gdf)}

    gdf = normalize_schema(gdf, epoch, access_date)
    gdf = validate_geometries(gdf)
    gdf = classify_boundary_status(gdf, india, METRIC_CRS)

    bc = gdf["boundary_status"].value_counts()
    stats["inside"] = int(bc.get("inside", 0))
    stats["boundary"] = int(bc.get("boundary", 0))
    stats["outside"] = int(bc.get("outside", 0))
    stats["india_candidates"] = stats["inside"] + stats["boundary"]

    # Keep India candidates only
    india_gdf = gdf[gdf["boundary_status"].isin(["inside", "boundary"])].copy()
    india_gdf = assign_states(india_gdf, india_states)
    india_gdf = validate_coordinates(india_gdf)
    india_gdf = detect_duplicates(india_gdf, METRIC_CRS)

    # Collect stats
    rc = india_gdf["geometry_repair_status"].value_counts()
    stats["valid_original"] = int(rc.get("valid_original", 0))
    stats["repaired"] = int(rc.get("repaired", 0))
    stats["unrepairable"] = int(rc.get("unrepairable", 0))

    dc = india_gdf["duplicate_status"].value_counts()
    stats["unique"] = int(dc.get("unique", 0))
    stats["exact_duplicates"] = int(dc.get("exact_duplicate", 0))
    stats["near_duplicate_candidates"] = int(dc.get("near_duplicate_candidate", 0))

    stats["final_count"] = len(india_gdf)

    for fld in (
        "elevation_m", "lake_area_km2", "lake_type",
        "glacier_id", "river_basin", "state_or_region",
    ):
        stats[f"null_{fld}"] = int(
            india_gdf[fld].isna().sum() if fld in india_gdf.columns
            else len(india_gdf)
        )

    return india_gdf, stats

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="GlacioGuard — PANGAEA glacial lake inventory ingestion"
    )
    ap.add_argument("--force", action="store_true",
                    help="Re-download all source data")
    args = ap.parse_args()

    access_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 60)
    print("  GlacioGuard — Glacial Lake Inventory Pipeline")
    print("=" * 60)
    print(f"  Source : PANGAEA Kumar & Vijay (2026)")
    print(f"  DOI   : https://doi.org/{PANGAEA_DOI}")
    print(f"  Date  : {access_date}")

    # ── Stage 1: Download ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 1 — Download")
    print("=" * 60)
    shp_2016 = download_pangaea_epoch("2016", args.force)
    shp_2022 = download_pangaea_epoch("2022", args.force)
    admin0_zip, admin1_zip = download_natural_earth(args.force)

    india, india_states = load_india_boundary(admin0_zip, admin1_zip)

    # ── Stage 2: Process 2022 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2 — Process 2022 (primary)")
    print("=" * 60)
    gdf_2022, stats_2022 = process_epoch(
        "2022", shp_2022, india, india_states, access_date
    )

    # ── Stage 3: Process 2016 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 3 — Process 2016 (historical)")
    print("=" * 60)
    gdf_2016, stats_2016 = process_epoch(
        "2016", shp_2016, india, india_states, access_date
    )

    # ── Stage 4: Assign UIDs ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 4 — Assign lake UIDs")
    print("=" * 60)
    gdf_2022, gdf_2016, match_stats = assign_lake_uids(
        gdf_2022, gdf_2016, METRIC_CRS
    )

    # ── Stage 5: Write outputs ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 5 — Write outputs")
    print("=" * 60)
    write_outputs(gdf_2022, "2022")
    write_outputs(gdf_2016, "2016")

    report_path = DATA_PROCESSED / "inventory_quality_report.md"
    generate_quality_report(
        {
            "access_date": access_date,
            "2022": stats_2022,
            "2016": stats_2016,
            "match_stats": match_stats,
        },
        report_path,
    )

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for epoch, st in [("2022", stats_2022), ("2016", stats_2016)]:
        print(f"\n  {epoch}:")
        print(f"    Source records        : {st['source_records']}")
        print(f"    India candidates     : {st['india_candidates']}")
        print(f"    Inside               : {st['inside']}")
        print(f"    Boundary             : {st['boundary']}")
        print(f"    Exact duplicates     : {st['exact_duplicates']}")
        print(f"    Near-dup candidates  : {st['near_duplicate_candidates']}")
        print(f"    Repaired geometries  : {st['repaired']}")
        print(f"    Unrepairable         : {st['unrepairable']}")
        print(f"    Final valid lakes    : {st['final_count']}")

    print(f"\n  Cross-epoch matching:")
    print(f"    2016 matched to 2022 : {match_stats['matched_2016']}")
    print(f"    2016 unmatched       : {match_stats['unmatched_2016']}")
    print(f"\n  Outputs written successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
