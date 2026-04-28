# SQL Parcel Join Workflow

Links HRWUI structure polygons to UGRC parcel boundaries via spatial join, then enriches with Utah Office of the State Auditor (OSA) 2025 tax assessment data. Supports HB48 fee administration by identifying structure ownership and property information within the HRWUI.

---

## Database

| Setting | Value |
|---|---|
| Server | sde1.csbs.utah.edu |
| Port | 5432 |
| Database | sdb_u0972368 |
| Extensions | postgis, plpgsql |
| Tools | PGAdmin 4, QGIS DB Manager |

---

## Tables Required

| Table | Source | Notes |
|---|---|---|
| `build_poly_w_vals` | FFSL/OMF statewide structures | Loaded via QGIS DB Manager |
| `[county]_parcels` | UGRC county parcel download | One per county - see instructions below |
| `osa_raw` | Utah OSA 2025 CSV (1.6M rows) | Loaded via PGAdmin Import/Export |

---

## Running for a New County

### Step 1 - Download county parcels from UGRC
Go to: https://gis.utah.gov/products/sgid/cadastre/parcels/  
Download the shapefile for your target county.

### Step 2 - Load into PostGIS via QGIS DB Manager
1. Open QGIS and add the county parcels shapefile as a layer
2. Open **Database → DB Manager**
3. Connect to `sdb_u0972368`
4. Go to **Table → Import Layer/File**
5. Set table name to `[county]_parcels` (e.g. `davis_parcels`)
6. Check **Create spatial index** and **Convert field names to lowercase**
7. Click OK

### Step 3 - Update the SQL script
Open `ffsl_parcel_join.sql` and update the four lines marked `← UPDATE`:
- Table names (`weber_parcels` → your county table)
- County name in OSA filter (`'Weber County'` → your county)
- Parcel ID cleaning rule (see `parcel_join_strategy.csv`)

### Step 4 - Run in PGAdmin
Open `ffsl_parcel_join.sql` in the PGAdmin query tool and run each step in order.

---

## Parcel ID Cleaning Rules

See `parcel_join_strategy.csv` for the full county-by-county analysis.

| Rule | Counties |
|---|---|
| `LTRIM(parcel_id, '0')` | Beaver, Carbon, Davis, Duchesne, Grand, Rich, Salt Lake, Uintah, Wayne, Weber |
| `LTRIM(REPLACE(parcel_id, '-', ''), '0')` | Tooele |
| Direct match (no cleaning) | Cache, Garfield, Juab, Kane, Millard, Sanpete, Sevier, Summit, Washington |
| Join via `serial_id_clean` | Morgan, Piute |
| No OSA data available | Box Elder, Daggett, Emery, Iron, Wasatch |
| No viable join | San Juan |

---

## Spatial Join Method

A structure is assigned to a parcel if **more than 50% of the structure's area** falls within the parcel boundary. All spatial operations are performed in EPSG:5070 (NAD83 Albers Equal Area, meters).

Parcel geometry is pre-transformed to EPSG:5070 and indexed once per county (Step 1 in the SQL script) to avoid expensive on-the-fly reprojection during the join.

---

## Output Fields

| Field | Source | Description |
|---|---|---|
| `structure_id` | Structures | Unique structure ID |
| `ses` | Structures | Structural Exposure Score |
| `parcel_id_raw` | UGRC | Raw parcel ID |
| `parcel_id_cleaned` | UGRC | Cleaned parcel ID (after county-specific rule) |
| `parcel_address` | UGRC | Parcel situs address |
| `owner_name` | OSA | Formatted owner name |
| `property_type` | OSA | Property type classification |
| `primary_residence` | OSA | Primary residence flag |
| `tax_exempt` | OSA | Tax exempt status |
| `year_built` | OSA | Year structure was built |
| `sq_feet` | OSA | Structure square footage |
| `market_value` | OSA | Assessed market value |
| `taxes_charged` | OSA | Annual taxes charged |
