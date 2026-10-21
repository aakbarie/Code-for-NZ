from pathlib import Path

import duckdb
import pandas as pd
import pytest

from morrill import db
from morrill.ingest import boreholes

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1.00E-96", "1E-96"), ("2.0E-97", "2E-97"), ("1-F-96", "1-F-96"), (" 1A-95 ", "1A-95")],
)
def test_demangle_hole_id(raw, expected):
    assert boreholes.demangle_hole_id(raw) == expected


def test_match_key_ignores_dashes_and_case():
    assert boreholes.match_key("1-F-96") == boreholes.match_key("1f-96") == "1F96"


def test_read_induction_logs_drops_side_table_and_placeholders():
    logs = boreholes.read_induction_logs(FIXTURES / "induction_log.csv")
    assert list(logs["hole_id"].unique()) == ["1E-96", "1-F-96"]
    assert len(logs) == 5
    assert logs.attrs["skipped_rows"] == 1
    assert logs["conductivity_ms_m"].notna().all()


def test_build_boreholes_merges_spellings():
    logs = boreholes.read_induction_logs(FIXTURES / "induction_log.csv")
    lith = boreholes.read_lithology(FIXTURES / "lithology.csv")
    holes = boreholes.build_boreholes(logs, lith).set_index("match_key")

    assert len(holes) == 3
    assert holes.loc["1E96", "hole_id"] == "1E-96"
    # collar elevation comes from the log (sample elevation + depth), not the lithology
    assert holes.loc["1E96", "collar_elev_m"] == pytest.approx(100.0)
    assert holes.loc["1F96", ["has_induction_log", "has_lithology"]].tolist() == [True, True]
    assert holes.loc["9Z99", ["has_induction_log", "has_lithology"]].tolist() == [False, True]


def test_build_boreholes_refuses_to_merge_distant_holes():
    logs = boreholes.read_induction_logs(FIXTURES / "induction_log.csv")
    lith = boreholes.read_lithology(FIXTURES / "lithology.csv")
    lith.loc[lith["hole_id"] == "1-F-96", "x"] += 500
    with pytest.raises(ValueError, match="1F96"):
        boreholes.build_boreholes(logs, lith)


def test_unexpected_mask_value_stops_the_load(tmp_path):
    text = (
        (FIXTURES / "lithology.csv")
        .read_text()
        .replace("Clay,None,9000.0,9000.0,200.0,1", "Clay,None,9000.0,9000.0,200.0,0")
    )
    (tmp_path / "lith.csv").write_text(text)
    with pytest.raises(ValueError, match="Mask"):
        boreholes.read_lithology(tmp_path / "lith.csv")


@pytest.fixture
def con():
    con = duckdb.connect()
    db.create_schema(con)
    db.load_lithology_classes(con)
    # two 2 m layers over a half-space
    con.execute("INSERT INTO model_layer VALUES (1, 2, 0, 2), (2, 2, 2, 4), (3, NULL, 4, NULL)")
    return con


def test_load_into_database(con):
    counts = boreholes.load(con, FIXTURES, "induction_log.csv", "lithology.csv")
    assert counts == {
        "induction_log_skipped": 1,
        "borehole": 3,
        "induction_log": 5,
        "lithology": 4,
    }
    names = {r[0] for r in con.sql("SELECT DISTINCT hole_id FROM induction_log").fetchall()}
    assert names == {"1E-96", "1-F-96"}


def test_layer_average_is_taken_in_conductivity(con):
    boreholes.load(con, FIXTURES, "induction_log.csv", "lithology.csv")
    row = con.sql(
        "SELECT resistivity_ohm_m, resistivity_naive_ohm_m, n_samples "
        "FROM borehole_layer_resistivity WHERE hole_id = '1E-96' AND layer = 1"
    ).fetchone()
    # samples of 10 and 1000 mS/m: mean conductivity 505 mS/m -> 1.98 ohm·m,
    # while averaging the resistivities (100 and 1 ohm·m) would give 50.5 ohm·m
    assert row[0] == pytest.approx(1000 / 505)
    assert row[1] == pytest.approx(50.5)
    assert row[2] == 2


def test_low_conductivity_flagged_unreliable(con):
    boreholes.load(con, FIXTURES, "induction_log.csv", "lithology.csv")
    unreliable = con.sql(
        "SELECT hole_id, depth_m FROM induction_log_clean WHERE NOT reliable"
    ).fetchall()
    assert unreliable == [("1-F-96", 2.0)]


def test_every_lithology_label_has_a_class(con):
    boreholes.load(con, FIXTURES, "induction_log.csv", "lithology.csv")
    unmapped = con.sql("SELECT * FROM lithology_classified WHERE texture IS NULL").fetchall()
    assert unmapped == []


def test_reference_classes_are_unique_and_valid():
    classes = pd.read_csv(
        Path(db.__file__).parent / "reference" / "lithology_classes.csv", dtype=str
    )
    assert classes["rock_raw"].is_unique
    con = duckdb.connect()
    db.create_schema(con)
    db.load_lithology_classes(con)  # CHECK constraints reject unknown classes
    assert con.sql("SELECT count(*) FROM lithology_class").fetchone()[0] == len(classes)
