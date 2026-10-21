"""Data checks run at the end of every build. Results go in the qc_result table.

A failed check doesn't stop the build: it marks something a person should look
at before trusting the affected rows.
"""

from collections.abc import Callable

import duckdb

Check = Callable[[duckdb.DuckDBPyConnection], tuple[bool, str]]


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    return con.sql(sql).fetchone()[0]


def regular_log_spacing(con):
    """Induction logs should be sampled every 0.1 ft (0.03048 m)."""
    gaps = _count(
        con,
        """
        SELECT count(*) FROM (
            SELECT depth_m - lag(depth_m) OVER (PARTITION BY hole_id ORDER BY depth_m) AS step
            FROM induction_log)
        WHERE step IS NOT NULL AND abs(step - 0.03048) > 0.001
        """,
    )
    return gaps == 0, f"{gaps} steps differ from 0.03048 m"


def _contiguous(table: str) -> Check:
    def check(con):
        rows = con.sql(
            f"""
            SELECT hole_id, round(coalesce(prev_to, 0), 2) AS gap_from, round(from_m, 2) AS gap_to
            FROM (
                SELECT hole_id, from_m,
                       lag(to_m) OVER (PARTITION BY hole_id ORDER BY from_m) AS prev_to
                FROM {table})
            WHERE abs(from_m - coalesce(prev_to, 0)) > 0.05
            ORDER BY hole_id, from_m
            """
        ).fetchall()
        return not rows, f"{len(rows)} gaps or overlaps (hole, from, to): {rows}"

    # 5 cm tolerance: depths were converted from feet and some were re-typed in
    # metres (e.g. 1M-97: 60.96 m = 200 ft, next interval starts at 61.0 m).
    check.__doc__ = f"{table} intervals in each hole should start at 0 m and meet end to end."
    return check


def rock_labels_classified(con):
    """Every driller's label needs a row in reference/lithology_classes.csv."""
    labels = [
        r[0]
        for r in con.sql(
            "SELECT DISTINCT rock_raw FROM lithology "
            "EXCEPT SELECT rock_raw FROM lithology_class ORDER BY 1"
        ).fetchall()
    ]
    return not labels, f"unclassified labels: {labels}" if labels else "all labels classified"


def collars_agree(con):
    """The files' collar elevations for the same hole should agree within 0.5 m."""
    rows = con.sql(
        "SELECT hole_id, round(collar_spread_m, 2) FROM borehole "
        "WHERE collar_spread_m > 0.5 ORDER BY collar_spread_m DESC"
    ).fetchall()
    return not rows, f"holes disagreeing by > 0.5 m: {rows}" if rows else "all within 0.5 m"


def layers_reach_125m(con):
    """The 24 inversion layers should add up to 125 m above the half-space."""
    total = _count(con, "SELECT sum(thickness_m) FROM model_layer")
    return abs(total - 125.0) < 0.01, f"layers sum to {total:.3f} m"


def few_unreliable_readings(con):
    """Readings below 1 mS/m are near the tool's noise floor; expect under 1%."""
    n, bad = con.sql(
        "SELECT count(*), count(*) FILTER (WHERE NOT reliable) FROM induction_log_clean"
    ).fetchone()
    share = bad / n if n else 0.0
    return share < 0.01, f"{bad} of {n} readings ({share:.2%}) below 1 mS/m"


def model_locations_matched(con):
    """Every model location should sit on an airborne sounding (within 1 m)."""
    total, unmatched = con.sql(
        "SELECT count(*), count(*) FILTER (WHERE sounding_id IS NULL) "
        "FROM (SELECT DISTINCT run_id, location_id, sounding_id FROM model_resistivity)"
    ).fetchone()
    return unmatched == 0, f"{unmatched} of {total} model locations not matched to a sounding"


def plausible_model_resistivity(con):
    """Inverted resistivities outside 1-1000 ohm·m are artefacts; expect under 1%."""
    n, bad = con.sql(
        "SELECT count(*), count(*) FILTER (WHERE NOT plausible) FROM model_resistivity_clean"
    ).fetchone()
    share = bad / n if n else 0.0
    return share < 0.01, f"{bad} of {n} values ({share:.2%}) outside 1-1000 ohm·m"


def datum_offset_consistent(con):
    """Per-hole (collar - DTM) offsets should agree: median absolute deviation under 1 m."""
    row = con.sql("SELECT offset_m, n_holes, spread_m FROM vertical_datum").fetchone()
    if row is None:
        return False, "no boreholes near enough to a sounding to estimate the offset"
    offset, n, spread = row
    return spread < 1.0, f"offset {offset:.2f} m from {n} holes, MAD {spread:.2f} m"


def bird_height_reasonable(con):
    """Above about 60 m the EM signal is weak; expect under 5% of soundings that high."""
    n, high = con.sql(
        "SELECT count(*), count(*) FILTER (WHERE bird_height_m > 60) FROM sounding"
    ).fetchone()
    share = high / n if n else 0.0
    return share < 0.05, f"{high} of {n} soundings ({share:.1%}) flown above 60 m"


CHECKS: dict[str, Check] = {
    "induction_log_regular_spacing": regular_log_spacing,
    "lithology_contiguous": _contiguous("lithology"),
    "stratigraphy_contiguous": _contiguous("stratigraphy"),
    "rock_labels_classified": rock_labels_classified,
    "collar_elevations_agree": collars_agree,
    "layers_sum_to_125m": layers_reach_125m,
    "few_unreliable_readings": few_unreliable_readings,
    "model_locations_matched": model_locations_matched,
    "plausible_model_resistivity": plausible_model_resistivity,
    "datum_offset_consistent": datum_offset_consistent,
    "bird_height_reasonable": bird_height_reasonable,
}


def run(con: duckdb.DuckDBPyConnection) -> list[tuple[str, bool, str]]:
    results = [(name, *check(con)) for name, check in CHECKS.items()]
    con.executemany("INSERT INTO qc_result VALUES (?, ?, ?)", results)
    return results
