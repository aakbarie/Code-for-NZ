# Data sources

Everything the database is built from, where it lives, what we know about it,
and what is still missing. The machine-readable list, with Drive IDs and
checksums, is [`manifests/raw_files.csv`](../manifests/raw_files.csv).

## Status

| Data | File | In the database? |
|---|---|---|
| Borehole induction logs | `CSD_ires.csv` | yes (32 holes, 37,394 readings) |
| Driller's lithology | `CSD_Lith.csv` | yes (130 holes, 1,173 intervals) |
| Formation picks | `CSD_strat.csv` | yes (102 holes, 210 intervals) |
| Inversion layering | `inv_model_pt1_bi.csv`, line 1 | yes (24 layers + half-space) |
| RESOLVE system | `hemSystem.txt` (New 07.16) | yes (`hem_channel`) |
| **Airborne survey** | `NPblock2008_phaseCorr_fixed.csv` (16 MB) | **no**: file not yet retrieved |
| **Inverted resistivity** | `inv_model_pt1.csv` (9.9 MB) | **no**: file not yet retrieved |
| 2015 SOM benchmark | `Morrill_block_test_*_voxler.csv` (9–12 MB each) | no: file not yet retrieved |

The large files are in Google Drive but couldn't be pulled into the build
environment: the Drive connector stops at about 5–10 MB per file. Download
them by hand into `data/raw/` at the paths in the manifest, then run
`uv run morrill verify-raw` and `uv run morrill build`.

## The airborne survey

- **Survey:** Fugro RESOLVE helicopter EM, flown June 15–23, 2008, for the
  North Platte Natural Resources District (Fugro report 08035).
- **Block:** the Morrill block is the survey's block 3 (lines `300xx`), with
  north–south lines about 200 m apart.
- **Master file:** `NPblock2008_phaseCorr_fixed.csv` has exactly 71,470 rows,
  the sounding count in the 2015 paper. That count is inferred from file
  sizes and needs checking once the file is local. The `_v1`, `_d1` and
  `_all_v1` variants add flag columns to the same rows.
- **Columns** (from a Drive search snippet): `Line`, `FID`, `x`, `y`, `DTM`,
  `LASER`, `Z`, `MAG`, `CPPL`, and the 12 EM channels `CPI400`, `CPQ400` …
  `CPQ140K`.
- **Coordinates:** NAD83, UTM zone 13N (EPSG:26913).
- **Heights are ellipsoidal.** The report gives the DTM "in metres above the
  WGS84 ellipsoid". Borehole collars sit 14.5–16.9 m above the nearby DTM,
  which is consistent with the geoid–ellipsoid separation; a geoid model
  (e.g. GEOID18) should confirm it. The airborne loader must convert heights
  before comparing them with boreholes.
- **Units:** EM channels are in ppm. The coplanar coils (CP) are 7.9 m apart;
  the coaxial coils (CX, 3,345 Hz) are 9.0 m apart.
- **Channel suffixes:** none = final levelled data, `_CAL` = raw,
  `_R` = background levelled.
- **Altitude:** `LASER` is the bird's height above ground (Optech laser
  altimeter).
- **Sampling:** 10 Hz, about 3.2 m between soundings.

## The inversion

- **Coverage:** `inv_model_pt1.csv` holds EM1DFM inversions for 41,330 of the
  71,470 soundings. No "part 2" was found.
- **Format:** 5 header lines, then per sounding x, y and 25 conductivities in
  S/m. The x and y values are stored at single precision (about 0.5 m), so
  match them to soundings with a tolerance.
- **Layering:** 24 layers over a half-space. The thicknesses run from 0.86 m
  to 15.53 m and add up to 125 m.
- **`inv_model_pt1_bi.csv`:** row-aligned with the inversion file, with a 1
  wherever resistivity is below **20 ohm·m**. That is the binary split the
  2015 paper's abstract describes. Its methods text and Table 6 say 5 ohm·m,
  which looks like a typo.

## Boreholes (Nebraska Conservation and Survey Division test holes)

- **Induction logs:** conductivity in mS/m every 0.1 ft. `RES_` is roughly
  1000/COND (98% of rows within 1%).
  - `CSD_iresSelectMorrill.csv` is an Excel re-save of the same data. It lost
    precision and turned three hole names into numbers. Don't use it.
- **`DH_RL`** is the elevation of the reading in the logs, but of the
  interval's **midpoint** in lithology and stratigraphy.
- **Collar elevations agree** within 0.5 m across all three files.
- **Known data problems** (kept as delivered and flagged by the build's checks):
  - **MO-15-02** (stratigraphy): a "Quaternary 0–0 m" interval followed by
    Brule from 27.4 m. The Quaternary probably runs 0–27.4 m.
  - **1M-97** (lithology): a 4 cm gap at 60.96 → 61.0 m, from rounding feet to
    metres.
  - **1N-97:** 18 m of "Clay" that logs at about 71 ohm·m, far above any other
    clay in the area. The label, the depth registration, or the clay itself is
    unusual.

## Not found

- **Burke Minsley's MCMC results.** Only one test sounding exists, and it's
  not from this survey. The run for line 30050 (7,110 soundings) was set up,
  but its output went to a local folder that isn't in Drive.
- **Raw water-quality data.** The 2015 paper used pore-water specific
  conductivity from about 22 wells. The only copy found sits inside 2012 SOM
  output, where gaps were filled by the model and measured values aren't
  marked, so it can't serve as ground truth.
