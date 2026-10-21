"""EM1DFM inversion output: x, y and layer conductivities for each sounding.

Format of inv_model_pt1.csv:
    line 1  ,Number,of,layers:,25
    line 2  ,Layer,thicknesses,(m):,<24 values>
    line 3  ,Number,of,soundings:,41330
    line 4  column description
    line 5  thicknesses again
    then    x, y, <25 conductivities in S/m>, (trailing comma)
x and y were written in single precision (about 0.5 m), so soundings are
matched by nearest location within a tolerance rather than exactly.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from morrill.db import insert_df, register_source


def _numbers(line: str) -> list[float]:
    out = []
    for field in line.strip().split(","):
        try:
            out.append(float(field))
        except ValueError:
            pass
    return out


def read_em1dfm(path: Path) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Return (layer thicknesses, xy array [n, 2], conductivity array [n, layers])."""
    with open(path, encoding="utf-8-sig") as fh:
        header = [next(fh) for _ in range(5)]
    n_layers = int(_numbers(header[0])[0])
    thicknesses = _numbers(header[1])
    n_soundings = int(_numbers(header[2])[0])

    data = np.genfromtxt(path, delimiter=",", skip_header=5)[:, : 2 + n_layers]
    if data.shape != (n_soundings, 2 + n_layers):
        raise ValueError(
            f"{path}: expected {n_soundings} x {2 + n_layers} values, got {data.shape}"
        )
    if len(thicknesses) != n_layers - 1:
        raise ValueError(f"{path}: {len(thicknesses)} thicknesses for {n_layers} layers")
    return thicknesses, data[:, :2], data[:, 2:]


def load(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    path: str,
    run_id: str = "em1dfm_pt1",
    tol_m: float = 1.0,
) -> dict[str, int]:
    thicknesses, xy, sigma = read_em1dfm(raw_dir / path)

    expected = [
        r[0] for r in con.sql("SELECT thickness_m FROM model_layer ORDER BY layer").fetchall()
    ]
    if not np.allclose(thicknesses, expected[:-1], rtol=1e-4) or len(expected) != sigma.shape[1]:
        raise ValueError(f"{path}: layering differs from model_layer; it needs its own layer table")

    register_source(con, run_id, raw_dir, path, "EM1DFM inversion, part 1")
    con.execute(
        "INSERT INTO model_run VALUES (?, 'EM1DFM 1D inversion', ?, ?)",
        [run_id, run_id, "Deterministic 1D inversion (UBC EM1DFM), 24 layers over a half-space"],
    )

    soundings = con.sql("SELECT sounding_id, x, y FROM sounding ORDER BY sounding_id").df()
    sounding_id = np.full(len(xy), np.nan)
    if len(soundings):
        dist, idx = cKDTree(soundings[["x", "y"]].to_numpy()).query(xy)
        matched = dist <= tol_m
        sounding_id[matched] = soundings["sounding_id"].to_numpy()[idx[matched]]

    n, k = sigma.shape
    rows = pd.DataFrame(
        {
            "run_id": run_id,
            "location_id": np.repeat(np.arange(1, n + 1), k),
            "layer": np.tile(np.arange(1, k + 1), n),
            "x": np.repeat(xy[:, 0], k),
            "y": np.repeat(xy[:, 1], k),
            "sounding_id": pd.array(np.repeat(sounding_id, k), dtype="Int64"),
            "resistivity_ohm_m": (1.0 / sigma).ravel(),
        }
    )
    return {
        "model_resistivity": insert_df(con, "model_resistivity", rows),
        "model_locations_unmatched": int(np.isnan(sounding_id).sum()),
    }
