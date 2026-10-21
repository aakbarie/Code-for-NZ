-- Morrill block database.
--
-- Conventions used in every table:
--   x, y            NAD83 / UTM zone 13N (EPSG:26913) easting / northing, metres
--   depth (_m)      metres below ground surface, positive down
--   elevation (_m)  metres above sea level (orthometric). The Fugro survey delivered
--                   heights above the ellipsoid, about 15 m lower; the airborne
--                   loader must convert them before they go in.
--   resistivity     ohm·m;  conductivity  mS/m  (resistivity = 1000 / conductivity)
-- Every loaded row can be traced to the raw file it came from (source_file).

CREATE TABLE source_file (
    source_id   VARCHAR PRIMARY KEY,   -- short name used by the loaders
    local_path  VARCHAR NOT NULL,      -- relative to data/raw
    sha256      VARCHAR NOT NULL,
    description VARCHAR
);

-- ---------------------------------------------------------------- boreholes

-- Nebraska Conservation and Survey Division (CSD) test holes.
CREATE TABLE borehole (
    hole_id           VARCHAR PRIMARY KEY,    -- CSD name, e.g. '1-F-96'
    match_key         VARCHAR NOT NULL UNIQUE,-- name without dashes; matches spellings across files
    x                 DOUBLE NOT NULL,
    y                 DOUBLE NOT NULL,
    collar_elev_m     DOUBLE,                 -- ground elevation at the hole (NAVD88, presumably)
    collar_spread_m   DOUBLE,                 -- disagreement between the files' collar elevations
    has_induction_log BOOLEAN NOT NULL,
    has_lithology     BOOLEAN NOT NULL,
    has_stratigraphy  BOOLEAN NOT NULL
);

-- Downhole induction logs, sampled every 0.1 ft (0.03048 m).
CREATE TABLE induction_log (
    hole_id               VARCHAR NOT NULL REFERENCES borehole (hole_id),
    depth_m               DOUBLE NOT NULL CHECK (depth_m >= 0),
    conductivity_ms_m     DOUBLE NOT NULL CHECK (conductivity_ms_m > 0),  -- what the tool measures
    resistivity_raw_ohm_m DOUBLE,   -- RES_ column as delivered; ~2% of rows disagree with 1000/conductivity
    las_file              VARCHAR,  -- name of the original LAS log file
    source_id             VARCHAR NOT NULL REFERENCES source_file (source_id),
    PRIMARY KEY (hole_id, depth_m)
);

-- Driller's lithology descriptions, as depth intervals.
CREATE TABLE lithology (
    hole_id    VARCHAR NOT NULL REFERENCES borehole (hole_id),
    from_m     DOUBLE NOT NULL CHECK (from_m >= 0),
    to_m       DOUBLE NOT NULL,
    rock_raw   VARCHAR NOT NULL,  -- label as delivered (cut off at 16 characters)
    drill_note VARCHAR,           -- e.g. 'Fast drill', 'Rig chatter'
    source_id  VARCHAR NOT NULL REFERENCES source_file (source_id),
    PRIMARY KEY (hole_id, from_m),
    CHECK (to_m > from_m)
);

-- Geologic units picked in each hole, from the youngest (top) down.
CREATE TABLE stratigraphy (
    hole_id    VARCHAR NOT NULL REFERENCES borehole (hole_id),
    from_m     DOUBLE NOT NULL CHECK (from_m >= 0),
    to_m       DOUBLE NOT NULL,
    system     VARCHAR NOT NULL,  -- Quaternary | Tertiary | Cretaceous
    series     VARCHAR,           -- Oligocene, Eocene, Miocene, ...
    group_name VARCHAR,           -- White River, Ogallala, Montana
    formation  VARCHAR,           -- Brule, Chadron, Ash Hollow, Pierre
    member     VARCHAR,           -- Orella, Whitney, ...
    source_id  VARCHAR NOT NULL REFERENCES source_file (source_id),
    PRIMARY KEY (hole_id, from_m),
    CHECK (to_m > from_m)
);

-- Reference table: what each driller's label means for hydrogeology.
-- Loaded from src/morrill/reference/lithology_classes.csv; edit that file to change it.
CREATE TABLE lithology_class (
    rock_raw     VARCHAR PRIMARY KEY,
    rock_full    VARCHAR NOT NULL,  -- label without the 16-character cut-off
    texture      VARCHAR NOT NULL CHECK (texture IN (
                     'clay', 'silt', 'silty_sand', 'sand', 'sand_gravel', 'gravel',
                     'mudrock', 'sandstone', 'ash', 'soil', 'fill', 'unknown')),
    consolidated BOOLEAN,           -- bedrock (true) vs loose sediment (false); NULL = transitional
    aquifer_role VARCHAR NOT NULL CHECK (aquifer_role IN ('aquifer', 'confining', 'unknown')),
    note         VARCHAR
);

-- ---------------------------------------------------------------- airborne

-- The six frequencies of the RESOLVE helicopter EM system.
CREATE TABLE hem_channel (
    channel          VARCHAR PRIMARY KEY,  -- e.g. 'CP400' (coplanar, 400 Hz)
    coil_orientation VARCHAR NOT NULL CHECK (coil_orientation IN ('coplanar', 'coaxial')),
    freq_nominal_hz  INTEGER NOT NULL,
    freq_hz          DOUBLE NOT NULL,      -- calibrated frequency
    coil_sep_m       DOUBLE NOT NULL,
    noise_ppm        DOUBLE                -- assumed data error (paper Table 1)
);

-- One row per airborne sounding (one sample along a flight line).
CREATE TABLE sounding (
    sounding_id       INTEGER PRIMARY KEY,   -- row order in the survey file, from 1
    line              INTEGER NOT NULL,      -- e.g. 30010; tie lines are 39xxx
    is_tie_line       BOOLEAN NOT NULL,
    fid               DOUBLE NOT NULL,       -- fiducial (time stamp, 10 per second) along the line
    x                 DOUBLE NOT NULL,
    y                 DOUBLE NOT NULL,
    dtm_ellipsoid_m   DOUBLE,  -- ground height as delivered: above the ellipsoid
    ground_elev_m     DOUBLE,  -- the same, shifted to the boreholes' datum (see vertical_datum)
    bird_height_m     DOUBLE,  -- laser altimeter: sensor height above ground
    powerline_monitor DOUBLE,  -- CPPL: high values mean power-line (cultural) noise
    mag_nt            DOUBLE,  -- magnetic field as delivered (looks like the residual field)
    source_id         VARCHAR NOT NULL REFERENCES source_file (source_id),
    UNIQUE (line, fid)
);

-- How airborne heights were moved onto the boreholes' vertical datum.
CREATE TABLE vertical_datum (
    name     VARCHAR PRIMARY KEY,
    offset_m DOUBLE NOT NULL,   -- added to ellipsoidal heights
    n_holes  INTEGER NOT NULL,  -- boreholes used to estimate it
    spread_m DOUBLE,            -- median absolute deviation of the per-hole offsets
    method   VARCHAR NOT NULL
);

-- The measured response at each sounding: in-phase (i) and quadrature (q), in ppm.
CREATE TABLE hem_obs (
    sounding_id INTEGER PRIMARY KEY REFERENCES sounding (sounding_id),
    cp400_i   DOUBLE, cp400_q   DOUBLE,
    cp1800_i  DOUBLE, cp1800_q  DOUBLE,
    cx3300_i  DOUBLE, cx3300_q  DOUBLE,
    cp8200_i  DOUBLE, cp8200_q  DOUBLE,
    cp40k_i   DOUBLE, cp40k_q   DOUBLE,
    cp140k_i  DOUBLE, cp140k_q  DOUBLE
);

-- ---------------------------------------------------------------- earth model

-- The fixed 1D layering used by the inversions: 24 layers over a half-space.
CREATE TABLE model_layer (
    layer       INTEGER PRIMARY KEY CHECK (layer >= 1),  -- 1 = top
    thickness_m DOUBLE,           -- NULL for the half-space
    top_m       DOUBLE NOT NULL,
    bottom_m    DOUBLE            -- NULL for the half-space
);

-- One row per way of producing a resistivity model (EM1DFM inversion, MCMC, 2015 SOM, ...).
CREATE TABLE model_run (
    run_id      VARCHAR PRIMARY KEY,
    method      VARCHAR NOT NULL,
    source_id   VARCHAR REFERENCES source_file (source_id),
    description VARCHAR
);

-- Layer resistivities from each run, located by the run's own x, y.
-- sounding_id links to the airborne sounding at the same place, when there is one.
CREATE TABLE model_resistivity (
    run_id            VARCHAR NOT NULL REFERENCES model_run (run_id),
    location_id       INTEGER NOT NULL,  -- row number within the run
    layer             INTEGER NOT NULL REFERENCES model_layer (layer),
    x                 DOUBLE NOT NULL,
    y                 DOUBLE NOT NULL,
    sounding_id       INTEGER REFERENCES sounding (sounding_id),
    resistivity_ohm_m DOUBLE CHECK (resistivity_ohm_m > 0),
    PRIMARY KEY (run_id, location_id, layer)
);

-- ---------------------------------------------------------------- links

-- Airborne soundings near each borehole (filled by the build, using a KD-tree).
CREATE TABLE borehole_sounding (
    hole_id     VARCHAR NOT NULL REFERENCES borehole (hole_id),
    sounding_id INTEGER NOT NULL REFERENCES sounding (sounding_id),
    distance_m  DOUBLE NOT NULL,
    rank        INTEGER NOT NULL,   -- 1 = nearest
    PRIMARY KEY (hole_id, sounding_id)
);

-- Results of the data checks run at the end of each build.
CREATE TABLE qc_result (
    check_name VARCHAR PRIMARY KEY,
    passed     BOOLEAN NOT NULL,
    detail     VARCHAR
);

-- ---------------------------------------------------------------- views

-- Induction resistivity with a reliability flag. Below ~1 mS/m the tool is near
-- its noise floor and the implied resistivity (>1000 ohm·m) can't be trusted.
CREATE VIEW induction_log_clean AS
SELECT
    hole_id,
    depth_m,
    conductivity_ms_m,
    1000.0 / conductivity_ms_m AS resistivity_ohm_m,
    conductivity_ms_m >= 1.0   AS reliable
FROM induction_log;

-- Each induction log averaged over each model layer, so boreholes can be compared
-- with airborne results layer by layer. Horizontal EM loops drive currents sideways
-- through the layers, so the right average is the mean CONDUCTIVITY (conductances
-- add in parallel). The naive mean of resistivity is kept to show the difference.
CREATE VIEW borehole_layer_resistivity AS
SELECT
    l.hole_id,
    m.layer,
    count(*)                                    AS n_samples,
    least(count(*) * 0.03048 / m.thickness_m, 1.0) AS coverage,  -- share of the layer the log covers
    1000.0 / avg(l.conductivity_ms_m)           AS resistivity_ohm_m,
    avg(1000.0 / l.conductivity_ms_m)           AS resistivity_naive_ohm_m,
    bool_and(l.conductivity_ms_m >= 1.0)        AS all_reliable
FROM induction_log AS l
JOIN model_layer AS m
  ON l.depth_m >= m.top_m
 AND (m.bottom_m IS NULL OR l.depth_m < m.bottom_m)
WHERE m.thickness_m IS NOT NULL
GROUP BY l.hole_id, m.layer, m.thickness_m;

-- Base of the surficial aquifer at each hole: the top of the first unit older than
-- the Quaternary (bedrock). A hole that never left the Quaternary only tells us the
-- base is deeper than the hole ("censored"): bedrock_depth_m is NULL there and the
-- base lies somewhere below hole_depth_m.
CREATE VIEW aquifer_base AS
SELECT
    b.hole_id,
    b.x,
    b.y,
    b.collar_elev_m,
    min(s.from_m) FILTER (WHERE s.system <> 'Quaternary')                 AS bedrock_depth_m,
    b.collar_elev_m - min(s.from_m) FILTER (WHERE s.system <> 'Quaternary') AS bedrock_elev_m,
    arg_min(coalesce(s.formation, s.group_name, s.system), s.from_m)
        FILTER (WHERE s.system <> 'Quaternary')                            AS bedrock_unit,
    bool_or(s.system <> 'Quaternary')                                      AS reached_bedrock,
    max(s.to_m)                                                            AS hole_depth_m
FROM borehole AS b
JOIN stratigraphy AS s USING (hole_id)
GROUP BY b.hole_id, b.x, b.y, b.collar_elev_m;

-- Model resistivities with a plausibility flag. Values outside 1-1000 ohm·m are
-- inversion artefacts here (the 2015 paper also dropped values above 1000 ohm·m).
CREATE VIEW model_resistivity_clean AS
SELECT *, resistivity_ohm_m BETWEEN 1 AND 1000 AS plausible
FROM model_resistivity;

-- Ground truth against each model run: the borehole log averaged over a layer
-- (in conductivity) next to the model's value for that layer at the nearest
-- sounding the run covers.
CREATE VIEW borehole_vs_model AS
WITH covered AS (
    SELECT DISTINCT run_id, sounding_id FROM model_resistivity WHERE sounding_id IS NOT NULL
),
nearest AS (
    SELECT bs.hole_id, c.run_id, bs.sounding_id, bs.distance_m,
           row_number() OVER (PARTITION BY bs.hole_id, c.run_id ORDER BY bs.distance_m) AS k
    FROM borehole_sounding AS bs
    JOIN covered AS c USING (sounding_id)
)
SELECT
    n.hole_id, n.run_id, n.sounding_id, n.distance_m,
    b.layer, b.coverage,
    b.resistivity_ohm_m                    AS log_resistivity_ohm_m,
    m.resistivity_ohm_m                    AS model_resistivity_ohm_m,
    m.resistivity_ohm_m BETWEEN 1 AND 1000 AS model_plausible
FROM nearest AS n
JOIN borehole_layer_resistivity AS b ON b.hole_id = n.hole_id
JOIN model_resistivity AS m
  ON m.run_id = n.run_id AND m.sounding_id = n.sounding_id AND m.layer = b.layer
WHERE n.k = 1;

-- Lithology intervals with their hydrogeologic class.
CREATE VIEW lithology_classified AS
SELECT
    l.hole_id, l.from_m, l.to_m, l.to_m - l.from_m AS thickness_m,
    l.rock_raw, c.rock_full, c.texture, c.consolidated, c.aquifer_role
FROM lithology AS l
LEFT JOIN lithology_class AS c USING (rock_raw);
