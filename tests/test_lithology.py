from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from morrill import db
from morrill.analysis import lithology as L
from morrill.ingest import airborne, boreholes, inversion, layers

FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_pairs() -> pd.DataFrame:
    """Coarse sediment near 60 ohm·m, bedrock near 6 ohm·m, a little fine near 15."""
    rows = []
    for hole in range(6):
        for layer, (log_rho, cls) in enumerate(
            [(1.8, "coarse"), (1.75, "coarse"), (1.2, "fine"), (0.8, "bedrock"), (0.75, "bedrock")],
            start=1,
        ):
            rows.append(
                {
                    "hole_id": f"h{hole}",
                    "sounding_id": hole,
                    "layer": layer,
                    "top_m": 4.0 * (layer - 1),
                    "bottom_m": 4.0 * layer,
                    "log_rho": log_rho + 0.02 * hole,
                    **{c: 4.0 * (c == cls) for c in L.CLASSES},
                }
            )
    return pd.DataFrame(rows)


def test_histogram_model_separates_classes():
    model = L.HistogramModel().fit(synthetic_pairs())
    proba = model.predict_cells(np.array([1.8, 0.8]), np.array([10.0, 10.0]))
    assert proba.argmax(axis=1).tolist() == [L.CLASSES.index("coarse"), L.BEDROCK]
    assert np.allclose(proba.sum(axis=1), 1)


def test_unseen_resistivity_falls_back_to_the_prior():
    model = L.HistogramModel(by_depth=False).fit(synthetic_pairs())
    proba = model.predict_cells(np.array([2.95]), np.array([10.0]))[0]
    assert proba == pytest.approx(model.prior_[0], abs=0.02)


def test_superposition_raises_bedrock_below_bedrock():
    pairs = synthetic_pairs()
    depth = np.array([2.0, 6.0, 10.0])
    profile = np.array([[1.8, 0.8, 1.4]])  # sand, bedrock, then an ambiguous value
    on = L.HistogramModel().fit(pairs).predict_profiles(profile, depth)[0]
    off = L.HistogramModel(superposition=False).fit(pairs).predict_profiles(profile, depth)[0]
    assert on[0] == pytest.approx(off[0]) and on[1] == pytest.approx(off[1])
    assert on[2, L.BEDROCK] > off[2, L.BEDROCK]


def test_prior_alone_ignores_resistivity():
    model = L.HistogramModel(use_resistivity=False).fit(synthetic_pairs())
    proba = model.predict_cells(np.array([0.8, 1.8]), np.array([10.0, 10.0]))
    assert np.allclose(proba[0], proba[1])


@pytest.fixture
def con():
    con = duckdb.connect()
    db.create_schema(con)
    db.load_reference(con)
    layers.load(con, FIXTURES, "layers.csv")
    boreholes.load(con, FIXTURES, "induction_log.csv", "lithology.csv", "stratigraphy.csv")
    airborne.load(con, FIXTURES, "survey.csv")
    inversion.load(con, FIXTURES, "inversion.csv")
    return con


def test_training_pairs_split_logged_metres_by_layer(con):
    pairs = L.training_pairs(con)
    got = {(r.hole_id, r.layer): (r.coarse, r.fine, r.bedrock) for r in pairs.itertuples()}
    # 1E-96: sand and gravel 0-2 m, Brule 2-5 m; 1-F-96: silt to siltstone 0-3 m.
    # 9-Z-99 is too far from any inverted sounding.
    assert got == {
        ("1E-96", 1): (2.0, 0.0, 0.0),
        ("1E-96", 2): (0.0, 0.0, 2.0),
        ("1-F-96", 1): (0.0, 2.0, 0.0),
        ("1-F-96", 2): (0.0, 1.0, 0.0),
    }
