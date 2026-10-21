import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from morrill.analysis import kalman


def synthetic_survey(n_lines=2, n=300, seed=1):
    """Geology drifts along each line; the pilot climbs and dives independently of it."""
    rng = np.random.default_rng(seed)
    rows = []
    sid = 1
    for line in range(n_lines):
        geology = np.cumsum(rng.normal(scale=0.05, size=n))
        height = 35 * np.exp(0.25 * np.sin(np.linspace(0, 8 + line, n)))
        for t in range(n):
            # every channel weakens with height; geology shifts channels by different amounts
            base = 3.0 - 2.0 * np.log(height[t] / 35) + geology[t] * np.linspace(-1, 1, 12)
            ppm = np.exp(base + rng.normal(scale=0.02, size=12))
            rows.append(
                {
                    "sounding_id": sid,
                    "line": 30010 + 10 * line,
                    "bird_height_m": height[t],
                    "geology": geology[t],
                    **dict(zip(kalman.CHANNEL_COLS, ppm, strict=True)),
                }
            )
            sid += 1
    return pd.DataFrame(rows)


def test_height_term_removes_flight_height_from_the_features():
    survey = synthetic_survey()
    lh = np.log(survey["bird_height_m"])
    with_h, *_ = kalman.extract(survey, k=1)
    without_h, *_ = kalman.extract(survey, k=1, use_height=False)
    r_with = abs(spearmanr(with_h["z1"], lh).statistic)
    r_without = abs(spearmanr(without_h["z1"], lh).statistic)
    assert r_with < 0.3 < r_without
    # and the feature tracks the geology
    assert abs(spearmanr(with_h["z1"], survey["geology"]).statistic) > 0.9


def test_features_have_one_row_per_sounding_with_uncertainty():
    survey = synthetic_survey()
    feats, model, history, _ = kalman.extract(survey, k=2)
    assert list(feats.index) == list(survey["sounding_id"])
    assert {"z1", "z2", "sd1", "sd2", "nis"} <= set(feats.columns)
    assert (feats[["sd1", "sd2"]] > 0).all().all()
    assert np.all(np.diff(history) > -1e-6 * np.abs(np.array(history[:-1])))
