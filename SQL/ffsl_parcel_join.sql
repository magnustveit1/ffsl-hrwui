-- ============================================================
-- FFSL HRWUI Structure → Parcel → OSA Join
-- Author: Magnus Tveit, University of Utah GIS Capstone
-- Last Updated: April 2026
--
-- Description:
--   Links HRWUI structure polygons to UGRC parcel boundaries
--   via spatial join, then enriches with Utah Office of the
--   State Auditor (OSA) 2025 tax assessment data.
--   Supports HB48 fee administration by identifying structure
--   ownership and property information within the HRWUI.
--
-- Database: sdb_u0972368 on sde1.csbs.utah.edu
-- Extensions required: postgis, plpgsql
--
-- ── INSTRUCTIONS FOR NEW COUNTY ──────────────────────────────
--   1. Download county parcels from UGRC:
--      https://gis.utah.gov/products/sgid/cadastre/parcels/
--   2. Load into PostGIS via QGIS DB Manager
--      Name the table: [county]_parcels  (e.g. davis_parcels)
--   3. Run Step 1 to pre-transform geometry (do once per county)
--   4. Update the four lines marked ← UPDATE in Step 2
--   5. Check parcel_join_strategy.csv for correct cleaning rule
--   6. Run Steps 2–4
-- ============================================================


-- ── STEP 1: Pre-transform parcel geometry to EPSG:5070 ───────
-- Run once per county after loading parcels.
-- Avoids expensive on-the-fly reprojection during spatial join.
-- Replace [county]_parcels with your county table name.

ALTER TABLE weber_parcels                          -- ← UPDATE
    ADD COLUMN IF NOT EXISTS geom_5070 geometry(Geometry, 5070);

UPDATE weber_parcels                               -- ← UPDATE
    SET geom_5070 = ST_Transform(geom, 5070)
    WHERE geom_5070 IS NULL;

CREATE INDEX IF NOT EXISTS weber_parcels_geom_5070_idx  -- ← UPDATE
    ON weber_parcels USING GIST(geom_5070);        -- ← UPDATE


-- ── STEP 2: Create county-specific structure subset ──────────
-- Pre-filters statewide structures to county bounding box.
-- Dramatically speeds up the spatial join in Step 3.
-- Replace weber_structures and weber_parcels as needed.

CREATE TABLE IF NOT EXISTS weber_structures AS     -- ← UPDATE
SELECT s.*
FROM build_poly_w_vals s
WHERE s.geom && (
    SELECT ST_Extent(ST_Transform(geom, 5070))
    FROM weber_parcels                             -- ← UPDATE
);

CREATE INDEX IF NOT EXISTS weber_structures_geom_idx  -- ← UPDATE
    ON weber_structures USING GIST(geom);         -- ← UPDATE


-- ── STEP 3: Full three-way join ───────────────────────────────
-- Structures → Parcels → OSA
-- Matching rule: structure must be >50% inside parcel (area-based)
-- Parcel ID cleaning rule for Weber: LTRIM(parcel_id, '0')
-- See parcel_join_strategy.csv for other county cleaning rules

SELECT
    -- Structure attributes
    s.id                            AS structure_id,
    s.ses                           AS structural_exposure_score,
    s.ses_rcl                       AS ses_reclassified,
    s.high_hazar                    AS high_hazard,
    s.wui_all                       AS wui_all,
    s.tom_wui                       AS tom_wui,

    -- Parcel attributes (from UGRC)
    w.parcel_id                     AS parcel_id_raw,
    LTRIM(w.parcel_id, '0')         AS parcel_id_cleaned,  -- ← UPDATE cleaning rule
    w.parcel_add                    AS parcel_address,
    w.parcel_cit                    AS parcel_city,
    w.parcel_zip                    AS parcel_zip,
    w.own_type                      AS ownership_type,

    -- Tax assessment attributes (from Utah OSA 2025)
    o.owner_name_formatted          AS owner_name,
    o.situs_address1                AS situs_address,
    o.property_type                 AS property_type,
    o.primary_residence             AS primary_residence,
    o.tax_exempt                    AS tax_exempt,
    o.tax_exempt_type               AS tax_exempt_type,
    o.year_built                    AS year_built,
    o.sq_feet                       AS sq_feet,
    o.construction_material         AS construction_material,
    o.floors_count                  AS floors,
    o.acres                         AS acres,
    o.market                        AS market_value,
    o.land                          AS land_value,
    o.improvements                  AS improvements_value,
    o.taxes_charged                 AS taxes_charged

FROM weber_structures s                            -- ← UPDATE

-- Spatial join: structure must be >50% inside parcel
JOIN weber_parcels w                               -- ← UPDATE
    ON ST_Intersects(s.geom, w.geom_5070)
    AND ST_Area(
        ST_Intersection(s.geom, w.geom_5070)
    ) / ST_Area(s.geom) > 0.5

-- Attribute join: cleaned parcel ID to OSA
JOIN osa_raw o
    ON LTRIM(w.parcel_id, '0') = o.parcel_id_clean  -- ← UPDATE cleaning rule
    AND o.county = 'Weber County'                    -- ← UPDATE county name

ORDER BY s.id;


-- ── STEP 4: Summary statistics ────────────────────────────────
-- Breakdown of matched structures by property type.
-- Useful for HB48 fee administration reporting.

SELECT
    o.property_type,
    COUNT(*)                            AS structure_count,
    COUNT(CASE WHEN o.tax_exempt = 'true'
               THEN 1 END)             AS tax_exempt_count,
    ROUND(AVG(o.sq_feet::numeric), 0)  AS avg_sq_feet,
    ROUND(AVG(o.taxes_charged::numeric),
          2)                           AS avg_taxes_charged
FROM weber_structures s                            -- ← UPDATE
JOIN weber_parcels w                               -- ← UPDATE
    ON ST_Intersects(s.geom, w.geom_5070)
    AND ST_Area(
        ST_Intersection(s.geom, w.geom_5070)
    ) / ST_Area(s.geom) > 0.5
JOIN osa_raw o
    ON LTRIM(w.parcel_id, '0') = o.parcel_id_clean -- ← UPDATE cleaning rule
    AND o.county = 'Weber County'                   -- ← UPDATE county name
GROUP BY o.property_type
ORDER BY structure_count DESC;
