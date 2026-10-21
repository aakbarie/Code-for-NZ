"""The helicopter EM survey: one row per sounding, plus its 12 measured channels.

The file is NPblock2008_phaseCorr_fixed.csv (Fugro RESOLVE survey, 2008,
block 3, phase-corrected in 2012). Quirks handled here:
- Missing values are written as '?' (the Weka convention).
- Heights (DTM, Z) are above the ellipsoid, not sea level. Borehole collars
  are on a sea-level (orthometric) datum, so we estimate the offset between
  the two from boreholes that sit next to a sounding, and store both heights.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from morrill.db import insert_df, register_source

# survey column -> hem_obs column
CHANNELS = {
    "CPI400": "cp400_i",
    "CPQ400": "cp400_q",
    "CPI1800": "cp1800_i",
    "CPQ1800": "cp1800_q",
    "CXI3300": "cx3300_i",
    "CXQ3300": "cx3300_q",
    "CPI8200": "cp8200_i",
    "CPQ8200": "cp8200_q",
    "CPI40K": "cp40k_i",
    "CPQ40K": "cp40k_q",
    "CPI140K": "cp140k_i",
    "CPQ140K": "cp140k_q",
}


def read_survey(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, na_values=["?"], dtype={"Line": str})
    # 'Line' is the name ('L30010', 'T39010'); 'LINE' the number. They must agree.
    if not (raw["Line"].str[1:] == raw["LINE"].astype(str)).all():
        raise ValueError(f"{path}: Line and LINE columns disagree")
    survey = pd.DataFrame(
        {
            "sounding_id": np.arange(1, len(raw) + 1),
            "line": raw["LINE"].astype(int),
            "is_tie_line": raw["Line"].str.startswith("T"),
            "fid": raw["FID"],
            "x": raw["x"],
            "y": raw["y"],
            "dtm_ellipsoid_m": raw["DTM"],
            "bird_height_m": raw["LASER"],
            "powerline_monitor": raw["CPPL"],
            "mag_nt": raw["MAG"],
        }
    )
    obs = raw[list(CHANNELS)].rename(columns=CHANNELS)
    return survey.join(obs)


def estimate_datum_offset(
    holes: pd.DataFrame, survey: pd.DataFrame, max_dist_m: float = 100.0
) -> tuple[float, int, float] | None:
    """Offset to add to ellipsoidal heights to match the boreholes' datum.

    Uses holes within max_dist_m of a sounding: offset = median of
    (collar elevation - DTM at the nearest sounding). The median resists the
    few holes where the ground changes height between hole and sounding.
    Returns (offset, holes used, median absolute deviation) or None.
    """
    dist, idx = cKDTree(survey[["x", "y"]].to_numpy()).query(holes[["x", "y"]].to_numpy())
    near = dist <= max_dist_m
    if not near.any():
        return None
    diffs = (
        holes["collar_elev_m"].to_numpy()[near] - survey["dtm_ellipsoid_m"].to_numpy()[idx[near]]
    )
    offset = float(np.median(diffs))
    return offset, int(near.sum()), float(np.median(np.abs(diffs - offset)))


def load(con: duckdb.DuckDBPyConnection, raw_dir: Path, path: str) -> dict[str, int]:
    """Load soundings and channels. Boreholes must already be loaded (for the datum offset)."""
    data = read_survey(raw_dir / path)
    register_source(con, "hem_survey", raw_dir, path, "Fugro RESOLVE 2008 survey, block 3")

    holes = con.sql("SELECT x, y, collar_elev_m FROM borehole WHERE collar_elev_m IS NOT NULL").df()
    datum = estimate_datum_offset(holes, data)
    if datum is not None:
        offset, n, spread = datum
        data["ground_elev_m"] = data["dtm_ellipsoid_m"] + offset
        con.execute(
            "INSERT INTO vertical_datum VALUES ('ellipsoid_to_borehole', ?, ?, ?, ?)",
            [offset, n, spread, "median(collar - DTM) at holes within 100 m of a sounding"],
        )
    else:
        data["ground_elev_m"] = np.nan

    sounding_cols = [
        "sounding_id", "line", "is_tie_line", "fid", "x", "y", "dtm_ellipsoid_m",
        "ground_elev_m", "bird_height_m", "powerline_monitor", "mag_nt",
    ]  # fmt: skip
    return {
        "sounding": insert_df(con, "sounding", data[sounding_cols].assign(source_id="hem_survey")),
        "hem_obs": insert_df(con, "hem_obs", data[["sounding_id", *CHANNELS.values()]]),
    }
