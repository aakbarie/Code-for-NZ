import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from morrill.analysis import bedrock, bedrock_profunctor, enki, fdem, profunctor
from morrill.analysis.profunctor import MAX_PLUS, SUM_PRODUCT


def random_line(rng, n=4, k=3):
    log_prior = np.log(rng.dirichlet(np.ones(k)))
    kernels = np.log(rng.dirichlet(np.ones(k), size=(n - 1, k)))
    liks = rng.normal(size=(n, k))
    return log_prior, kernels, liks


def all_paths(log_prior, kernels, liks):
    n, k = liks.shape
    paths = np.array(list(itertools.product(range(k), repeat=n)))
    score = log_prior[paths[:, 0]] + liks[np.arange(n), paths].sum(axis=1)
    for t in range(n - 1):
        score += kernels[t][paths[:, t], paths[:, t + 1]]
    return paths, score


@pytest.mark.parametrize("semiring", [SUM_PRODUCT, MAX_PLUS])
def test_composition_is_associative_with_an_identity(semiring):
    rng = np.random.default_rng(0)
    P, Q, R = (rng.normal(size=(3, 3)) for _ in range(3))
    left = profunctor.compose(profunctor.compose(P, Q, semiring), R, semiring)
    right = profunctor.compose(P, profunctor.compose(Q, R, semiring), semiring)
    np.testing.assert_allclose(left, right)
    identity = np.where(np.eye(3, dtype=bool), 0.0, -np.inf)
    np.testing.assert_allclose(profunctor.compose(P, identity, semiring), P)


def test_sum_product_along_a_line_is_the_sum_over_every_path():
    rng = np.random.default_rng(1)
    log_prior, kernels, liks = random_line(rng)
    paths, score = all_paths(log_prior, kernels, liks)
    log_post, evidence = profunctor.posterior(log_prior, kernels, liks)
    assert evidence == pytest.approx(logsumexp(score))
    for t in range(liks.shape[0]):
        exact = [logsumexp(score[paths[:, t] == s]) - evidence for s in range(liks.shape[1])]
        np.testing.assert_allclose(log_post[t], exact)


def test_max_plus_along_a_line_is_the_best_path():
    rng = np.random.default_rng(2)
    log_prior, kernels, liks = random_line(rng, n=5)
    paths, score = all_paths(log_prior, kernels, liks)
    np.testing.assert_array_equal(
        profunctor.best_path(log_prior, kernels, liks), paths[score.argmax()]
    )
    assert profunctor.chain(log_prior, kernels, liks, MAX_PLUS)[2] == pytest.approx(score.max())


def test_the_line_kernel_keeps_the_prior_exactly():
    x = np.linspace(-4, 4, 60)
    log_prior = profunctor.grid_log_prior(x, 0.3, 1.2)
    for a in (0.5, 0.98, 0.9999):
        K = profunctor.reversible_kernel(x, log_prior, 0.3, 1.2, a)
        np.testing.assert_allclose(logsumexp(K, axis=1), 0.0, atol=1e-12)
        np.testing.assert_allclose(logsumexp(log_prior[:, None] + K, axis=0), log_prior, atol=1e-10)


def test_a_linear_gaussian_line_matches_the_exact_kalman_smoother():
    rng = np.random.default_rng(3)
    n, a, q_sd, r_sd = 30, 0.9, 1.0, 0.5
    truth = np.zeros(n)
    truth[0] = rng.normal()
    for t in range(1, n):
        truth[t] = a * truth[t - 1] + np.sqrt(1 - a**2) * rng.normal()
    y = truth + r_sd * rng.normal(size=n)
    # exact: Kalman filter and RTS smoother for x_t = a x_{t-1} + w, y_t = x_t + v
    m, P = np.zeros(n), np.zeros(n)
    mp, Pp = 0.0, q_sd**2
    for t in range(n):
        if t:
            mp, Pp = a * m[t - 1], a**2 * P[t - 1] + (1 - a**2) * q_sd**2
        gain = Pp / (Pp + r_sd**2)
        m[t], P[t] = mp + gain * (y[t] - mp), (1 - gain) * Pp
    ms, Ps = m.copy(), P.copy()
    for t in range(n - 2, -1, -1):
        Pp = a**2 * P[t] + (1 - a**2) * q_sd**2
        J = a * P[t] / Pp
        ms[t] = m[t] + J * (ms[t + 1] - a * m[t])
        Ps[t] = P[t] + J**2 * (Ps[t + 1] - Pp)
    x = np.linspace(-6, 6, 481)
    log_prior = profunctor.grid_log_prior(x, 0.0, q_sd)
    K = np.array([profunctor.reversible_kernel(x, log_prior, 0.0, q_sd, a)] * (n - 1))
    log_post, _ = profunctor.posterior(log_prior, K, -0.5 * ((y[:, None] - x) / r_sd) ** 2)
    p = np.exp(log_post)
    mean = p @ x
    sd = np.sqrt(p @ x**2 - mean**2)
    np.testing.assert_allclose(mean, ms, atol=0.01)
    np.testing.assert_allclose(sd, np.sqrt(Ps), atol=0.01)


def test_sampled_paths_follow_the_posterior():
    rng = np.random.default_rng(4)
    log_prior, kernels, liks = random_line(rng, n=4, k=3)
    log_post, _ = profunctor.posterior(log_prior, kernels, liks)
    fwd, _, _ = profunctor.chain(log_prior, kernels, liks)
    draws = profunctor.sample_paths(fwd, kernels, rng, 40_000)
    for t in range(4):
        freq = np.bincount(draws[t], minlength=3) / draws.shape[1]
        np.testing.assert_allclose(freq, np.exp(log_post[t]), atol=0.01)


def test_quantiles_and_answers_on_a_grid():
    x = np.linspace(-5, 5, 401)
    one = profunctor.grid_log_prior(x, 0.0, 1.0)[None]
    np.testing.assert_allclose(profunctor.quantiles(one, x)[:, 0], [-1.2816, 0, 1.2816], atol=0.01)
    two = np.logaddexp(
        profunctor.grid_log_prior(x, -2.0, 0.4), profunctor.grid_log_prior(x, 2.0, 0.4)
    )[None]
    assert profunctor.n_modes(one)[0] == 1 and profunctor.n_modes(two)[0] == 2


def test_laplace_is_exact_for_a_linear_forward_model():
    rng = np.random.default_rng(5)
    p, n, B = len(bedrock_profunctor.NU), 12, 3
    A = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    m = rng.normal(size=p)
    L = rng.normal(size=(p, p)) * 0.3
    P = L @ L.T + np.eye(p) * 0.5
    exact = (
        -0.5 * (y - A @ m) @ np.linalg.solve(np.eye(n) + A @ P @ A.T, y - A @ m)
        - 0.5 * np.linalg.slogdet(np.eye(n) + A @ P @ A.T)[1]
    )  # log ∫ N(ψ; m, P) exp(-|Aψ - y|²/2) dψ

    def resid(psi, log_z):
        return psi @ A.T - y

    _, ev, _ = bedrock_profunctor.laplace(
        resid, np.zeros((B, p)), np.zeros(B), m, np.linalg.inv(P), np.linalg.slogdet(P)[1],
        np.full(p, -1e9), np.full(p, 1e9), n_iter=3,
    )  # fmt: skip
    np.testing.assert_allclose(ev, exact, rtol=1e-6)


def test_a_deep_sharp_contact_is_found_on_the_grid():
    """The same synthetic as test_bedrock: 150 ohm·m gravel on 8 ohm·m shale at 45 m."""
    system = fdem.System(
        freq_hz=np.array([378.0, 1776.0, 3345.0, 8171.0, 41020.0, 129550.0]),
        coil_sep_m=np.array([7.93, 7.91, 9.03, 7.91, 7.91, 7.89]),
        coaxial=np.array([False, False, True, False, False, False]),
        noise_ppm=np.array([5.0, 10.0, 10.0, 10.0, 20.0, 40.0]),
    )
    stats = {
        "log_depth_mean": 0.77, "log_depth_sd": 0.47,
        "fill_mean": 1.65, "fill_sd": 0.25, "rock_mean": 1.2, "rock_sd": 0.25,
    }  # fmt: skip
    n, depth = 5, 45.0
    rng = np.random.default_rng(0)
    clean = fdem.channels(
        fdem.response(system, np.tile([1 / 150, 1 / 8], (n, 1)), np.full((n, 1), depth), 35.0)
    )
    sd = enki.data_error(clean, system.channel_noise, 0.05)
    ld = enki.LineData(
        line=1, sounding_id=np.arange(n), xy=np.zeros((n, 2)),
        d_obs=clean + sd * rng.standard_normal(clean.shape), height_m=np.full(n, 35.0),
        step_m=np.r_[0.0, np.full(n - 1, 64.0)], system=system,
        thickness=np.array([]), tops=np.array([0.0]),
    )  # fmt: skip
    settings = bedrock_profunctor.Settings(n_samples=20)
    prior = bedrock.make_prior(stats, bedrock.Settings(depth_sd_scale=1.0))
    out = bedrock_profunctor.invert_line(ld, prior, settings, seed=0)
    assert np.all(out["depth_q10"] <= depth) and np.all(out["depth_q90"] >= depth)
    assert np.all(np.abs(out["depth_q50"] / depth - 1) < 0.25)
    assert out["members"].shape == (n, 20)
