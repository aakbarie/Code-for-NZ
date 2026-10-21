import numpy as np
import pytest

from morrill.analysis import enki, fdem
from morrill.analysis import statespace as ss


def linear_prior(p=3, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(p, p))
    return enki.Prior(rng.normal(size=p), L @ L.T + p * np.eye(p), {})


def test_es_mda_gives_the_exact_posterior_for_a_linear_problem():
    rng = np.random.default_rng(1)
    prior = linear_prior()
    G = rng.normal(size=(4, 3))
    y = rng.normal(size=4)
    theta = prior.mean + rng.standard_normal((20000, 3)) @ prior.chol.T
    post, _ = enki.es_mda(theta, y, lambda t: t @ G.T, rng, n_assim=4, bounded=False)
    # textbook Gaussian posterior with R = I
    cov = np.linalg.inv(np.linalg.inv(prior.cov) + G.T @ G)
    mean = cov @ (np.linalg.solve(prior.cov, prior.mean) + G.T @ y)
    assert post.mean(axis=0) == pytest.approx(mean, abs=0.03)
    assert np.cov(post.T) == pytest.approx(cov, abs=0.05)


def test_forecast_keeps_the_prior_at_every_step():
    rng = np.random.default_rng(2)
    prior = linear_prior()
    theta = prior.mean + rng.standard_normal((40000, 3)) @ prior.chol.T
    for _ in range(5):
        theta = enki.forecast(theta, prior, a=0.8, rng=rng)
    assert theta.mean(axis=0) == pytest.approx(prior.mean, abs=0.05)
    assert np.cov(theta.T) == pytest.approx(prior.cov, rel=0.05, abs=0.05)


@pytest.mark.parametrize("n_assim", [1, 3])
def test_ensemble_filter_and_smoother_match_the_exact_kalman_smoother(n_assim):
    """Linear observations and AR(1) dynamics: the ensemble answer must be note 05's."""
    rng = np.random.default_rng(3)
    p, m, n, a = 2, 3, 40, 0.9
    prior = enki.Prior(np.zeros(p), np.array([[1.0, 0.5], [0.5, 2.0]]), {})
    C = rng.normal(size=(m, p))
    x, Y = rng.multivariate_normal(np.zeros(p), prior.cov), []
    for _ in range(n):
        x = a * x + rng.multivariate_normal(np.zeros(p), (1 - a * a) * prior.cov)
        Y.append(C @ x + rng.normal(size=m))
    Y = np.array(Y)
    observed = np.ones(n, dtype=bool)
    observed[15:20] = False  # a gap, as where the bird flies too high

    settings = enki.Settings(n_ensemble=4000, n_assim=n_assim, length_m=-1 / np.log(a))
    forwards = [lambda t: t @ C.T] * n
    F, A, _ = enki.filter_line(
        Y, forwards, np.ones(n), observed, prior, settings, rng, bounded=False
    )
    S = enki.rts_smoother(F, A)

    exact = ss.StateSpace(
        A=a * np.eye(p), C=C, D=np.zeros((m, 1)), Q=(1 - a * a) * prior.cov, R=np.ones(m),
        mu0=prior.mean, P0=prior.cov,
    )  # fmt: skip
    f = ss.kalman_filter(exact, Y, np.zeros((n, 1)), observed)
    xs, Ps, _ = ss.rts_smoother(exact, f)
    sd = np.sqrt(np.einsum("tii->ti", Ps))
    assert np.abs(A.mean(axis=1) - f.x_filt).max() < 0.1
    assert np.abs(S.mean(axis=1) - xs).max() < 0.1
    assert np.abs(S.std(axis=1) / sd - 1).max() < 0.1


def test_conductor_depth_is_the_first_conductive_layer_below_1_8_m():
    tops = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
    log_rho = np.log10(
        [
            [10, 10, 50, 15, 5],  # 10 ohm·m at 0-2 m is too shallow to count: conductor at 5 m
            [100, 100, 100, 100, 100],  # none
        ]
    )
    assert enki.conductor_depth(log_rho, tops).tolist() == [5.0, np.inf]


def test_prior_correlation_has_a_whole_profile_part_and_a_local_part():
    mid = np.array([1.0, 3.0, 50.0])
    prior = enki.make_prior(mid, mean=1.5, sd=0.3, c=0.4, length_m=5.0)
    corr = prior.cov[:3, :3] / 0.09
    assert corr[0, 1] == pytest.approx(0.4 + 0.6 * np.exp(-2 / 5))
    assert corr[0, 2] == pytest.approx(0.4, abs=1e-3)  # far apart: only the shared part
    assert prior.cov[3, 3] == enki.HEIGHT_SD**2 and prior.cov[0, 3] == 0
    np.linalg.cholesky(prior.cov)  # a valid covariance


def test_a_buried_conductor_is_found_with_an_honest_interval():
    """Synthetic line: 150 ohm·m gravel over an 8 ohm·m shale at 31.8 m, the bird at 35 m."""
    system = fdem.System(
        freq_hz=np.array([378.0, 1776.0, 3345.0, 8171.0, 41020.0, 129550.0]),
        coil_sep_m=np.array([7.93, 7.91, 9.03, 7.91, 7.91, 7.89]),
        coaxial=np.array([False, False, True, False, False, False]),
        noise_ppm=np.array([5.0, 10.0, 10.0, 10.0, 20.0, 40.0]),
    )
    thickness = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.3, 9.0, 12.0])
    tops = np.r_[0.0, np.cumsum(thickness)]  # the shale starts at the top of layer 9: 31.8 m
    truth = np.where(tops < 31.8, 150.0, 8.0)
    rng = np.random.default_rng(4)
    n = 6
    clean = fdem.channels(fdem.response(system, 1 / np.tile(truth, (n, 1)), thickness, 35.0))
    sd = enki.data_error(clean, system.channel_noise, 0.05)
    d_obs = clean + sd * rng.standard_normal(clean.shape)

    ld = enki.LineData(
        line=1, sounding_id=np.arange(n), xy=np.zeros((n, 2)), d_obs=d_obs,
        height_m=np.full(n, 35.0), step_m=np.r_[0.0, np.full(n - 1, 64.0)], system=system,
        thickness=thickness, tops=tops,
    )  # fmt: skip
    mid = tops + np.r_[thickness, 10.0] / 2
    prior = enki.make_prior(mid, mean=1.5, sd=0.5, c=0.4, length_m=5.0)
    out = enki.invert_line(ld, prior, enki.Settings(n_ensemble=100), seed=0)["coupled"]
    assert np.all(out["depth_q10"] <= 31.8) and np.all(out["depth_q90"] >= 31.8)
    assert np.nanmedian(np.mean(out["residual"] ** 2, axis=1)) < 2.0
