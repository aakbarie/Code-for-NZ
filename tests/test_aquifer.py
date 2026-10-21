import numpy as np
import pandas as pd

from morrill.analysis import aquifer, bedrock


def test_strip_weights_add_up_to_the_line_length():
    y = np.array([0.0, 50.0, 130.0, 200.0])
    w = aquifer.strip_weights(y)
    np.testing.assert_allclose(w, [25.0, 65.0, 75.0, 35.0])
    assert w.sum() == 200.0


def test_recalibrating_members_matches_widening_the_interval():
    rng = np.random.default_rng(0)
    members = 10 ** rng.normal(1.3, 0.1, size=(3, 4001))
    out = aquifer.recalibrate(members, 2.0)
    q10, q50, q90 = np.quantile(members, [0.1, 0.5, 0.9], axis=1)
    lo, hi = bedrock.widen(q10, q50, q90, 2.0)
    np.testing.assert_allclose(np.quantile(out, [0.1, 0.9], axis=1), [lo, hi], rtol=1e-3)


def test_a_flat_aquifer_gives_its_thickness_and_volume():
    n, N = 11, 50
    lines = list(aquifer.NS_LINES)
    steps = pd.DataFrame({
        "line": np.repeat(lines, n), "y": np.tile(np.linspace(0, 1000, n), len(lines)),
        "x": np.repeat(np.arange(len(lines)) * 200.0, n), "depth_q50": 10.0,
    })  # fmt: skip
    members = np.full((len(steps), N), 10.0)
    flat = np.full(len(steps), 10.0)
    table, draws = aquifer.thickness_table(steps, members, flat, flat, k=2.0)
    np.testing.assert_allclose(table[["sharp_median", "sharp_cal_q50", "smooth", "em1dfm"]], 10.0)
    widths = aquifer.strip_widths(steps.groupby("line")["x"].median())
    vol = aquifer.volumes(table, draws, widths)
    # 1 km of line by the 1,800 m between the outermost lines, 10 m thick
    np.testing.assert_allclose(vol.loc["sharp, raw", "q50"], 1000 * 1800 * 10 / 1e9)
