"""Links between boreholes and airborne soundings."""

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from morrill.db import insert_df


def link_boreholes(
    con: duckdb.DuckDBPyConnection, k: int = 10, max_dist_m: float = 1000.0
) -> dict[str, int]:
    """Store up to k nearest soundings within max_dist_m of each borehole."""
    soundings = con.sql("SELECT sounding_id, x, y FROM sounding ORDER BY sounding_id").df()
    holes = con.sql("SELECT hole_id, x, y FROM borehole ORDER BY hole_id").df()
    if soundings.empty or holes.empty:
        return {"borehole_sounding": 0}

    k = min(k, len(soundings))
    dist, idx = cKDTree(soundings[["x", "y"]].to_numpy()).query(holes[["x", "y"]].to_numpy(), k=k)
    dist, idx = dist.reshape(len(holes), k), idx.reshape(len(holes), k)
    links = pd.DataFrame(
        {
            "hole_id": np.repeat(holes["hole_id"].to_numpy(), k),
            "sounding_id": soundings["sounding_id"].to_numpy()[idx.ravel()],
            "distance_m": dist.ravel(),
            "rank": np.tile(np.arange(1, k + 1), len(holes)),
        }
    )
    return {
        "borehole_sounding": insert_df(
            con, "borehole_sounding", links[links["distance_m"] <= max_dist_m]
        )
    }
