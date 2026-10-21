"""CSD test holes: induction logs and driller's lithology.

Quirks in the delivered files that this module handles:
- Hole names are spelled differently across files ('1-F-96' vs '1F-96'), and
  Excel turned some into numbers ('1E-96' became '1.00E-96').
- The induction-log CSV has a second, unrelated table pasted into the columns
  to the right of 'Mask' on its first few dozen rows.
- Rock names are cut off at 16 characters; see reference/lithology_classes.csv.
"""

import csv
import re
from pathlib import Path

import duckdb
import pandas as pd

from morrill.db import insert_df, register_source

# '1.00E-96' -> '1E-96'
_EXCEL_SCIENTIFIC = re.compile(r"^(\d+)\.0+E-(\d+)$", re.IGNORECASE)


def demangle_hole_id(raw: str) -> str:
    """Undo Excel's conversion of hole names like '1E-96' into '1.00E-96'."""
    name = raw.strip()
    m = _EXCEL_SCIENTIFIC.match(name)
    return f"{m[1]}E-{m[2]}" if m else name


def match_key(hole_id: str) -> str:
    """Spelling-independent key: '1-F-96' and '1F-96' both become '1F96'."""
    return re.sub(r"[\s-]", "", hole_id).upper()


def _check_mask(df: pd.DataFrame, path: Path) -> None:
    # Every row in the files seen so far has Mask = 1. A different value would
    # mean rows were meant to be excluded, so stop and look rather than guess.
    if not (df["Mask"].str.strip() == "1").all():
        raise ValueError(f"{path}: unexpected Mask values {sorted(df['Mask'].unique())}")


def read_induction_logs(path: Path) -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = csv.reader(fh)
        header = next(rows)
        n = header.index("Mask") + 1  # columns after Mask are the pasted side table
        data = [row[:n] for row in rows if row and row[0].strip()]
    raw = pd.DataFrame(data, columns=header[:n])
    _check_mask(raw, path)
    logs = pd.DataFrame(
        {
            "hole_id_raw": raw["Line"].str.strip(),
            "hole_id": raw["Line"].map(demangle_hole_id),
            "depth_m": pd.to_numeric(raw["DH_Depth"]),
            "conductivity_ms_m": pd.to_numeric(raw["COND"]),
            "resistivity_raw_ohm_m": pd.to_numeric(raw["RES_"]),
            "las_file": raw["LAS"].str.strip().replace("", None),
            "x": pd.to_numeric(raw["DH_East"]),
            "y": pd.to_numeric(raw["DH_North"]),
            "elev_m": pd.to_numeric(raw["DH_RL"]),  # elevation of the sample
        }
    )
    # Placeholder rows carry a hole name but no measurement; skip and count them.
    empty = logs[["depth_m", "conductivity_ms_m"]].isna().any(axis=1)
    logs = logs[~empty].reset_index(drop=True)
    logs.attrs["skipped_rows"] = int(empty.sum())
    return logs


def read_lithology(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    _check_mask(raw, path)
    return pd.DataFrame(
        {
            "hole_id_raw": raw["Line"].str.strip(),
            "hole_id": raw["Line"].map(demangle_hole_id),
            "from_m": pd.to_numeric(raw["DH_From"]),
            "to_m": pd.to_numeric(raw["DH_To"]),
            "rock_raw": raw["ROCK"].str.strip(),
            "drill_note": raw["Notes"].str.strip().replace({"None": None, "": None}),
            "x": pd.to_numeric(raw["DH_East"]),
            "y": pd.to_numeric(raw["DH_North"]),
            "elev_m": pd.to_numeric(raw["DH_RL"]),  # elevation at the top of the interval
        }
    )


def build_boreholes(logs: pd.DataFrame, lith: pd.DataFrame, tol_m: float = 1.0) -> pd.DataFrame:
    """One row per hole, merging the spellings used by the two files.

    Spellings are matched with match_key, then confirmed by location: if two
    rows share a key but sit more than tol_m apart, they are different holes
    and we stop rather than merge them.
    """
    sites = pd.concat(
        [
            # ground elevation = elevation of a sample + its depth
            logs.assign(source="log", collar=logs["elev_m"] + logs["depth_m"]),
            lith.assign(source="lith", collar=lith["elev_m"] + lith["from_m"]),
        ]
    )
    sites["match_key"] = sites["hole_id"].map(match_key)

    spread = sites.groupby("match_key").agg(
        dx=("x", lambda s: s.max() - s.min()), dy=("y", lambda s: s.max() - s.min())
    )
    clash = spread[(spread["dx"] > tol_m) | (spread["dy"] > tol_m)]
    if not clash.empty:
        raise ValueError(f"same hole name at different places: {list(clash.index)}")

    def pick(group: pd.DataFrame) -> pd.Series:
        from_lith = group[group["source"] == "lith"]
        from_log = group[group["source"] == "log"]
        # Lithology spellings are the clean CSD names; prefer them for display.
        # Prefer the log's collar elevation: it is derived from many samples.
        name = (from_lith if not from_lith.empty else from_log)["hole_id"].iloc[0]
        collar = (from_log if not from_log.empty else from_lith)["collar"].median()
        return pd.Series(
            {
                "hole_id": name,
                "x": group["x"].mean(),
                "y": group["y"].mean(),
                "collar_elev_m": collar,
                "has_induction_log": not from_log.empty,
                "has_lithology": not from_lith.empty,
            }
        )

    holes = sites.groupby("match_key").apply(pick, include_groups=False).reset_index()
    return holes[
        ["hole_id", "match_key", "x", "y", "collar_elev_m", "has_induction_log", "has_lithology"]
    ]


def load(
    con: duckdb.DuckDBPyConnection, raw_dir: Path, log_path: str, lith_path: str
) -> dict[str, int]:
    """Load holes, logs and lithology. Paths are relative to raw_dir."""
    logs = read_induction_logs(raw_dir / log_path)
    lith = read_lithology(raw_dir / lith_path)
    holes = build_boreholes(logs, lith)

    register_source(con, "csd_induction", raw_dir, log_path, "CSD test-hole induction logs")
    register_source(con, "csd_lithology", raw_dir, lith_path, "CSD test-hole lithology")

    to_name = dict(zip(holes["match_key"], holes["hole_id"], strict=True))
    logs["hole_id"] = logs["hole_id"].map(match_key).map(to_name)
    lith["hole_id"] = lith["hole_id"].map(match_key).map(to_name)

    return {
        "induction_log_skipped": logs.attrs["skipped_rows"],
        "borehole": insert_df(con, "borehole", holes),
        "induction_log": insert_df(
            con,
            "induction_log",
            logs[
                ["hole_id", "depth_m", "conductivity_ms_m", "resistivity_raw_ohm_m", "las_file"]
            ].assign(source_id="csd_induction"),
        ),
        "lithology": insert_df(
            con,
            "lithology",
            lith[["hole_id", "from_m", "to_m", "rock_raw", "drill_note"]].assign(
                source_id="csd_lithology"
            ),
        ),
    }
