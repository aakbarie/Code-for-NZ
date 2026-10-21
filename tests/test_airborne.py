from pathlib import Path

import duckdb
import numpy as np
import pytest

from morrill import db, link
from morrill.ingest import airborne, boreholes, inversion, layers

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_survey_handles_question_marks_and_tie_lines():
    survey = airborne.read_survey(FIXTURES / "survey.csv")
    assert len(survey) == 4
    assert survey["sounding_id"].tolist() == [1, 2, 3, 4]
    assert np.isnan(survey.loc[1, "powerline_monitor"]) and np.isnan(survey.loc[1, "mag_nt"])
    assert survey["is_tie_line"].tolist() == [False, False, False, True]
    assert survey.loc[0, "cp400_i"] == 145


def test_read_survey_rejects_inconsistent_line_columns(tmp_path):
    text = (FIXTURES / "survey.csv").read_text().replace("L30020,", "L30030,", 1)
    (tmp_path / "s.csv").write_text(text)
    with pytest.raises(ValueError, match="disagree"):
        airborne.read_survey(tmp_path / "s.csv")


def test_datum_offset_uses_only_nearby_holes():
    survey = airborne.read_survey(FIXTURES / "survey.csv")
    logs = boreholes.read_induction_logs(FIXTURES / "induction_log.csv")
    lith = boreholes.read_lithology(FIXTURES / "lithology.csv")
    holes = boreholes.build_boreholes(logs, lith)
    # 1E-96 (collar 100) is 10 m from DTM 85, 1-F-96 (collar 150) 50 m from DTM 135;
    # 9-Z-99 is 700 m from the nearest sounding and is ignored.
    assert airborne.estimate_datum_offset(holes, survey) == (15.0, 2, 0.0)


def test_read_em1dfm():
    thick, xy, sigma = inversion.read_em1dfm(FIXTURES / "inversion.csv")
    assert thick == [2.0, 2.0]
    assert xy.shape == (4, 2) and sigma.shape == (4, 3)
    assert sigma[2, 1] == pytest.approx(1e-6)


@pytest.fixture
def con():
    con = duckdb.connect()
    db.create_schema(con)
    db.load_reference(con)
    layers.load(con, FIXTURES, "layers.csv")
    boreholes.load(con, FIXTURES, "induction_log.csv", "lithology.csv", "stratigraphy.csv")
    airborne.load(con, FIXTURES, "survey.csv")
    return con


def test_survey_loaded_with_corrected_heights(con):
    rows = con.sql(
        "SELECT sounding_id, dtm_ellipsoid_m, ground_elev_m FROM sounding ORDER BY 1"
    ).fetchall()
    assert rows[0] == (1, 85.0, 100.0)
    assert con.sql("SELECT offset_m, n_holes FROM vertical_datum").fetchone() == (15.0, 2)
    assert con.sql("SELECT count(*) FROM hem_obs").fetchone()[0] == 4


def test_inversion_matched_to_soundings(con):
    counts = inversion.load(con, FIXTURES, "inversion.csv")
    assert counts == {"model_resistivity": 12, "model_locations_unmatched": 1}
    rows = con.sql(
        "SELECT location_id, sounding_id, resistivity_ohm_m FROM model_resistivity "
        "WHERE layer = 1 ORDER BY location_id"
    ).fetchall()
    assert rows == [(1, 1, 100.0), (2, 2, 50.0), (3, 3, 10.0), (4, None, 20.0)]
    implausible = con.sql(
        "SELECT location_id, layer FROM model_resistivity_clean WHERE NOT plausible"
    ).fetchall()
    assert implausible == [(3, 2)]  # 1e-6 S/m = 1,000,000 ohm·m


def test_inversion_with_other_layering_is_refused(con, tmp_path):
    text = (FIXTURES / "inversion.csv").read_text().replace("(m):,2,2", "(m):,2,3")
    (tmp_path / "inv.csv").write_text(text)
    with pytest.raises(ValueError, match="layering"):
        inversion.load(con, tmp_path, "inv.csv")


def test_link_and_compare_with_boreholes(con):
    inversion.load(con, FIXTURES, "inversion.csv")
    link.link_boreholes(con, k=2, max_dist_m=1000)
    nearest = con.sql(
        "SELECT hole_id, sounding_id, round(distance_m, 1) FROM borehole_sounding "
        "WHERE rank = 1 ORDER BY hole_id"
    ).fetchall()
    assert nearest == [("1-F-96", 3, 50.0), ("1E-96", 1, 10.0), ("9-Z-99", 4, 707.1)]

    row = con.sql(
        "SELECT sounding_id, log_resistivity_ohm_m, model_resistivity_ohm_m "
        "FROM borehole_vs_model WHERE hole_id = '1E-96' AND layer = 1"
    ).fetchone()
    assert row == (1, pytest.approx(1000 / 505), 100.0)
