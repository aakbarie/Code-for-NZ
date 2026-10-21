"""CSD test holes: induction logs, driller's lithology, and formation picks.

Quirks in the delivered files that this module handles:
- Hole names are spelled differently across files ('1-F-96' vs '1F-96'), and
  an Excel re-save turned some into numbers ('1E-96' became '1.00E-96').
- That re-saved induction-log file also has a second, unrelated table pasted
  into the columns to the right of 'Mask' on its first few dozen rows.
- DH_RL means different things: in the induction logs it is the elevation of
  the reading; in lithology and stratigraphy it is the elevation of the MIDDLE
  of each interval.
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


def _blank_to_none(s: pd.Series) -> pd.Series:
    return s.str.strip().replace({"": None, "None": None})


def read_induction_logs(path: Path) -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = csv.reader(fh)
        header = next(rows)
        n = header.index("Mask") + 1  # anything after Mask is a pasted side table
        data = [row[:n] for row in rows if row and row[0].strip()]
    raw = pd.DataFrame(data, columns=header[:n])
    _check_mask(raw, path)
    logs = pd.DataFrame(
        {
            "hole_id": raw["Line"].map(demangle_hole_id),
            "depth_m": pd.to_numeric(raw["DH_Depth"]),
            "conductivity_ms_m": pd.to_numeric(raw["COND"]),
            "resistivity_raw_ohm_m": pd.to_numeric(raw["RES_"]),
            "las_file": _blank_to_none(raw["LAS"]),
            "x": pd.to_numeric(raw["DH_East"]),
            "y": pd.to_numeric(raw["DH_North"]),
            "collar_elev_m": pd.to_numeric(raw["DH_RL"]) + pd.to_numeric(raw["DH_Depth"]),
        }
    )
    # Placeholder rows carry a hole name but no measurement; skip and count them.
    empty = logs[["depth_m", "conductivity_ms_m"]].isna().any(axis=1)
    logs = logs[~empty].reset_index(drop=True)
    logs.attrs["skipped_rows"] = int(empty.sum())
    return logs


def _read_intervals(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Common part of the lithology and stratigraphy files: hole, interval, location."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    _check_mask(raw, path)
    from_m = pd.to_numeric(raw["DH_From"])
    to_m = pd.to_numeric(raw["DH_To"])
    intervals = pd.DataFrame(
        {
            "hole_id": raw["Line"].map(demangle_hole_id),
            "from_m": from_m,
            "to_m": to_m,
            "x": pd.to_numeric(raw["DH_East"]),
            "y": pd.to_numeric(raw["DH_North"]),
            # DH_RL is the elevation of the interval's midpoint
            "collar_elev_m": pd.to_numeric(raw["DH_RL"]) + (from_m + to_m) / 2,
        }
    )
    return raw, intervals


def _drop_empty_intervals(df: pd.DataFrame) -> pd.DataFrame:
    # e.g. hole MO-15-02 has 'Quaternary 0-0 m' followed by Brule from 27.4 m;
    # the true interval is probably 0-27.4 m, but we skip and count rather than guess.
    empty = df["to_m"] <= df["from_m"]
    out = df[~empty].reset_index(drop=True)
    out.attrs["skipped_rows"] = int(empty.sum())
    return out


def read_lithology(path: Path) -> pd.DataFrame:
    raw, lith = _read_intervals(path)
    lith["rock_raw"] = raw["ROCK"].str.strip()
    lith["drill_note"] = _blank_to_none(raw["Notes"])
    return _drop_empty_intervals(lith)


def read_stratigraphy(path: Path) -> pd.DataFrame:
    raw, strat = _read_intervals(path)
    for col, name in [
        ("Sys", "system"),
        ("Ser", "series"),
        ("Grp", "group_name"),
        ("Frm", "formation"),
        ("Mem", "member"),
    ]:
        strat[name] = _blank_to_none(raw[col])
    return _drop_empty_intervals(strat)


def build_boreholes(
    logs: pd.DataFrame,
    lith: pd.DataFrame,
    strat: pd.DataFrame | None = None,
    tol_m: float = 1.0,
) -> pd.DataFrame:
    """One row per hole, merging the spellings used by the different files.

    Spellings are matched with match_key, then confirmed by location: if two
    rows share a key but sit more than tol_m apart, they are different holes
    and we stop rather than merge them.
    """
    parts = [logs.assign(source="log"), lith.assign(source="lith")]
    if strat is not None:
        parts.append(strat.assign(source="strat"))
    sites = pd.concat(parts)[["hole_id", "x", "y", "collar_elev_m", "source"]]
    sites["match_key"] = sites["hole_id"].map(match_key)

    spread = sites.groupby("match_key").agg(
        dx=("x", lambda s: s.max() - s.min()), dy=("y", lambda s: s.max() - s.min())
    )
    clash = spread[(spread["dx"] > tol_m) | (spread["dy"] > tol_m)]
    if not clash.empty:
        raise ValueError(f"same hole name at different places: {list(clash.index)}")

    def pick(group: pd.DataFrame) -> pd.Series:
        sources = set(group["source"])
        # Lithology spellings are the clean CSD names; prefer them for display.
        for preferred in ("lith", "strat", "log"):
            if preferred in sources:
                name = group.loc[group["source"] == preferred, "hole_id"].iloc[0]
                break
        collar = group["collar_elev_m"]
        return pd.Series(
            {
                "hole_id": name,
                "x": group["x"].mean(),
                "y": group["y"].mean(),
                "collar_elev_m": collar.median(),
                "collar_spread_m": collar.max() - collar.min(),
                "has_induction_log": "log" in sources,
                "has_lithology": "lith" in sources,
                "has_stratigraphy": "strat" in sources,
            }
        )

    holes = sites.groupby("match_key").apply(pick, include_groups=False).reset_index()
    return holes[
        [
            "hole_id",
            "match_key",
            "x",
            "y",
            "collar_elev_m",
            "collar_spread_m",
            "has_induction_log",
            "has_lithology",
            "has_stratigraphy",
        ]
    ]


def load(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
    log_path: str,
    lith_path: str,
    strat_path: str | None = None,
) -> dict[str, int]:
    """Load holes, logs, lithology and (optionally) stratigraphy. Paths are relative to raw_dir."""
    logs = read_induction_logs(raw_dir / log_path)
    lith = read_lithology(raw_dir / lith_path)
    strat = read_stratigraphy(raw_dir / strat_path) if strat_path else None
    holes = build_boreholes(logs, lith, strat)

    register_source(con, "csd_induction", raw_dir, log_path, "CSD test-hole induction logs")
    register_source(con, "csd_lithology", raw_dir, lith_path, "CSD test-hole lithology")
    if strat_path:
        register_source(con, "csd_strat", raw_dir, strat_path, "CSD test-hole formation picks")

    to_name = dict(zip(holes["match_key"], holes["hole_id"], strict=True))

    def canonical(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(hole_id=df["hole_id"].map(match_key).map(to_name))

    counts = {
        "induction_log_skipped": logs.attrs["skipped_rows"],
        "lithology_skipped": lith.attrs["skipped_rows"],
        "borehole": insert_df(con, "borehole", holes),
        "induction_log": insert_df(
            con,
            "induction_log",
            canonical(logs)[
                ["hole_id", "depth_m", "conductivity_ms_m", "resistivity_raw_ohm_m", "las_file"]
            ].assign(source_id="csd_induction"),
        ),
        "lithology": insert_df(
            con,
            "lithology",
            canonical(lith)[["hole_id", "from_m", "to_m", "rock_raw", "drill_note"]].assign(
                source_id="csd_lithology"
            ),
        ),
    }
    if strat is not None:
        counts["stratigraphy_skipped"] = strat.attrs["skipped_rows"]
        counts["stratigraphy"] = insert_df(
            con,
            "stratigraphy",
            canonical(strat)[
                [
                    "hole_id",
                    "from_m",
                    "to_m",
                    "system",
                    "series",
                    "group_name",
                    "formation",
                    "member",
                ]
            ].assign(source_id="csd_strat"),
        )
    return counts
