import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from morrill.analysis import bedrock_field, bedrock_profunctor, profunctor


def tiny_field(rng, k=4, n=3):
    """Two lines of n steps on a grid of k depths, linked step by step across."""
    x = np.linspace(0.0, 1.5, k)
    lines = []
    for r in range(2):
        log_prior = profunctor.grid_log_prior(x, 0.7, 0.5)
        K = np.array([profunctor.reversible_kernel(x, log_prior, 0.7, 0.5, 0.8)] * (n - 1))
        lines.append(
            bedrock_field.Line(np.arange(n) + r * n, rng.normal(size=(n, k)), K, log_prior)
        )
    i, j = np.arange(n), np.arange(n) + n
    var = np.full(n, 0.2)
    edges = bedrock_field.Edges(np.r_[i, j], np.r_[j, i], np.r_[var, var])
    return bedrock_field.Field(lines, edges, x, 2 * n), lines, x, var


def exact_marginals(lines, x, var, n, k):
    states = np.array(list(itertools.product(range(k), repeat=2 * n)))
    score = np.zeros(len(states))
    for r, ln in enumerate(lines):
        z = states[:, r * n : (r + 1) * n]
        score += ln.log_prior[z[:, 0]] + ln.log_L[np.arange(n), z].sum(axis=1)
        for t in range(n - 1):
            score += ln.log_kernels[t][z[:, t], z[:, t + 1]]
    for t in range(n):
        score += -0.5 * (x[states[:, t]] - x[states[:, t + n]]) ** 2 / var[t]
    p = np.exp(score - logsumexp(score))
    return np.array([[p[states[:, s] == v].sum() for v in range(k)] for s in range(2 * n)])


def test_line_block_gibbs_converges_to_the_exact_2d_posterior():
    rng = np.random.default_rng(0)
    k, n = 4, 3
    field, lines, x, var = tiny_field(rng, k, n)
    exact = exact_marginals(lines, x, var, n, k)
    log_marg, kept, _ = field.gibbs(
        np.zeros((2 * n, k)), np.zeros(2 * n, dtype=int), 4000, 200, rng
    )
    np.testing.assert_allclose(np.exp(log_marg), exact, atol=0.015)
    freq = np.array([np.bincount(kept[s], minlength=k) / kept.shape[1] for s in range(2 * n)])
    np.testing.assert_allclose(freq, exact, atol=0.03)


def test_best_field_is_a_fixed_point_of_line_by_line_best_paths():
    rng = np.random.default_rng(1)
    field, lines, x, var = tiny_field(rng)
    z = field.icm(np.zeros((6, 4)), np.zeros(6, dtype=int), 20)
    for li, ln in enumerate(lines):
        best = profunctor.best_path(
            ln.log_prior, ln.log_kernels, field.unary(li, z, np.zeros((6, 4)))
        )
        np.testing.assert_array_equal(z[ln.rows], best)


def test_links_are_symmetric_and_local():
    y = np.arange(0, 1000, 64.0)
    steps = pd.DataFrame({
        "line": np.r_[np.full(len(y), 1), np.full(len(y), 2), np.full(4, 39001)],
        "x": np.r_[np.zeros(len(y)), np.full(len(y), 200.0), [-50, 0, 200, 250]],
        "y": np.r_[y, y + 10, np.full(4, 500.0)],
    })  # fmt: skip
    e = bedrock_field.across_edges(steps, [1, 2], [39001], 1e-5, bedrock_field.Settings())
    pairs = set(zip(e.i, e.j, strict=True))
    assert all((j, i) in pairs for i, j in pairs)
    line = steps["line"].to_numpy()
    assert np.all(line[e.i] != line[e.j])  # links only join different lines
    d = np.hypot(*(steps[["x", "y"]].to_numpy()[e.i] - steps[["x", "y"]].to_numpy()[e.j]).T)
    assert d.max() <= 300.0 and np.allclose(e.var, 1e-5 * d)


def test_the_hole_variogram_and_point_predictive():
    x = np.linspace(0, 2.2, 400)
    row = profunctor.grid_log_prior(x, 1.0, 0.05)
    pred = bedrock_field.point_predictive(row, x, 0.12**2)
    p = np.exp(pred - logsumexp(pred))
    mean = p @ x
    assert np.sqrt(p @ x**2 - mean**2) == pytest.approx(np.hypot(0.05, 0.12), rel=0.02)


def test_the_hole_error_is_recovered():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 2.2, 400)
    n, s0, v = 400, 0.15, 1e-4
    truth = rng.uniform(0.3, 1.8, n)
    log_marg = np.log(np.exp(-0.5 * ((x[None] - truth[:, None]) / 0.05) ** 2))  # the EM's posterior
    log_marg -= logsumexp(log_marg, axis=1, keepdims=True)
    d = rng.uniform(0, 100, n)
    holes = pd.DataFrame({
        "sounding_id": np.arange(n), "distance_m": d,
        "bedrock_depth_m": 10 ** (truth + rng.normal(0, np.sqrt(s0**2 + 0.05**2 + v * d))),
    })  # fmt: skip
    got, _ = bedrock_field.fit_hole_error(holes, dict(enumerate(range(n))), log_marg, x, v)
    assert got == pytest.approx(s0, abs=0.03)


def test_coarse_thickness_adds_up_the_fill_layers():
    from morrill.analysis import bedrock, coarse

    class AllCoarse:  # a lithology model that calls every layer coarse
        def predict_cells(self, log_rho, depth):
            return np.tile([1.0, 0.0, 0.0], (len(log_rho), 1))

    x = np.linspace(0, 2, 5)
    psi = np.zeros((2, 5, len(bedrock_profunctor.NU)))
    nodes = np.array([[0, 4], [2, 2]])
    out = coarse.coarse_thickness(AllCoarse(), psi, x, nodes)
    np.testing.assert_allclose(out, 10.0 ** x[nodes])  # all of the fill is coarse
    assert coarse.BOUNDS[-1] == pytest.approx(1.0) and bedrock.K == len(coarse.BOUNDS) - 1
