"""The fixed layering of the 1D inversions: 24 layers over a half-space.

The exact thicknesses are on the first line of inv_model_pt1_bi.csv
(the paper's Table 2 rounds them to two decimals).
"""

from itertools import accumulate
from pathlib import Path

import duckdb
import pandas as pd

from morrill.db import insert_df, register_source


def read_thicknesses(path: Path) -> list[float]:
    with open(path, encoding="utf-8-sig") as fh:
        first = fh.readline()
    return [float(v) for v in first.strip().rstrip(",").split(",")]


def layer_table(thicknesses: list[float]) -> pd.DataFrame:
    tops = [0.0, *accumulate(thicknesses)]
    n = len(thicknesses)
    return pd.DataFrame(
        {
            "layer": range(1, n + 2),
            "thickness_m": [*thicknesses, None],  # last row is the half-space
            "top_m": tops,
            "bottom_m": [*tops[1:], None],
        }
    )


def load(con: duckdb.DuckDBPyConnection, raw_dir: Path, path: str) -> dict[str, int]:
    register_source(con, "inv_layers", raw_dir, path, "Layer thicknesses of the 1D inversions")
    return {
        "model_layer": insert_df(con, "model_layer", layer_table(read_thicknesses(raw_dir / path)))
    }
