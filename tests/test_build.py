from pathlib import Path

import duckdb
import pytest

from morrill import build
from morrill.ingest import layers

FIXTURES = Path(__file__).parent / "fixtures"
SOURCES = {
    "induction": "induction_log.csv",
    "lithology": "lithology.csv",
    "stratigraphy": "stratigraphy.csv",
    "layers": "layers.csv",
}


def test_layer_table_adds_half_space():
    table = layers.layer_table([0.86, 0.98])
    assert table["layer"].tolist() == [1, 2, 3]
    assert table["top_m"].tolist() == pytest.approx([0.0, 0.86, 1.84])
    assert table["bottom_m"].tolist()[:2] == pytest.approx([0.86, 1.84])
    assert table["thickness_m"].isna().tolist() == [False, False, True]


def test_build_from_fixtures(tmp_path):
    db_path = tmp_path / "test.duckdb"
    results = build.build(db_path, FIXTURES, SOURCES)

    con = duckdb.connect(str(db_path), read_only=True)
    assert con.sql("SELECT count(*) FROM borehole").fetchone()[0] == 3
    assert con.sql("SELECT count(*) FROM hem_channel").fetchone()[0] == 6
    assert con.sql("SELECT count(*) FROM model_layer").fetchone()[0] == 3
    stored = con.sql("SELECT check_name, passed FROM qc_result ORDER BY 1").fetchall()
    assert stored == sorted((name, passed) for name, passed, _ in results)

    by_name = {name: passed for name, passed, _ in results}
    assert by_name["rock_labels_classified"]
    assert by_name["collar_elevations_agree"]
    assert by_name["lithology_contiguous"]
    # the fixture logs are deliberately irregular and the layers sum to 4 m
    assert not by_name["induction_log_regular_spacing"]
    assert not by_name["layers_sum_to_125m"]
    assert not (tmp_path / "test.duckdb.building").exists()


def test_build_stops_on_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="verify-raw"):
        build.build(tmp_path / "x.duckdb", tmp_path, SOURCES)
    assert not (tmp_path / "x.duckdb").exists()
