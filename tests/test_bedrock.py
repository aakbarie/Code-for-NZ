import numpy as np

from morrill.analysis import bedrock, enki, fdem

STATS = {  # roughly what the Morrill boreholes give
    "log_depth_mean": 0.77, "log_depth_sd": 0.47,
    "fill_mean": 1.65, "fill_sd": 0.25, "rock_mean": 1.2, "rock_sd": 0.25,
}  # fmt: skip
SYSTEM = fdem.System(
    freq_hz=np.array([378.0, 1776.0, 3345.0, 8171.0, 41020.0, 129550.0]),
    coil_sep_m=np.array([7.93, 7.91, 9.03, 7.91, 7.91, 7.89]),
    coaxial=np.array([False, False, True, False, False, False]),
    noise_ppm=np.array([5.0, 10.0, 10.0, 10.0, 20.0, 40.0]),
)


def test_each_member_has_its_own_contact_depth():
    theta = np.zeros((2, bedrock.K + 3))
    theta[:, bedrock.DEPTH] = np.log10([10.0, 40.0])
    sigma, thickness = bedrock.earth(theta)
    assert sigma.shape == (2, bedrock.K + 1) and thickness.shape == (2, bedrock.K)
    np.testing.assert_allclose(thickness.sum(axis=1), [10.0, 40.0])


def test_clip_keeps_the_contact_between_1_and_150_m():
    theta = np.zeros((2, bedrock.K + 3))
    theta[:, bedrock.DEPTH] = [-1.0, 4.0]
    out = bedrock.clip(theta.copy())
    np.testing.assert_allclose(10 ** out[:, bedrock.DEPTH], [1.0, 150.0])


def test_prior_is_a_valid_gaussian_with_the_borehole_depths():
    prior = bedrock.make_prior(STATS, bedrock.Settings(depth_sd_scale=2.0))
    np.linalg.cholesky(prior.cov)
    assert prior.mean[bedrock.DEPTH] == STATS["log_depth_mean"]
    assert np.sqrt(prior.cov[bedrock.DEPTH, bedrock.DEPTH]) == 2.0 * STATS["log_depth_sd"]


def test_a_deep_sharp_contact_is_found_despite_a_shallow_prior():
    """150 ohm·m gravel on 8 ohm·m shale at 45 m; the prior's median contact is at 6 m."""
    n, depth = 6, 45.0
    rng = np.random.default_rng(0)
    clean = fdem.channels(
        fdem.response(SYSTEM, np.tile([1 / 150, 1 / 8], (n, 1)), np.full((n, 1), depth), 35.0)
    )
    sd = enki.data_error(clean, SYSTEM.channel_noise, 0.05)
    ld = enki.LineData(
        line=1, sounding_id=np.arange(n), xy=np.zeros((n, 2)),
        d_obs=clean + sd * rng.standard_normal(clean.shape), height_m=np.full(n, 35.0),
        step_m=np.r_[0.0, np.full(n - 1, 64.0)], system=SYSTEM,
        thickness=np.array([]), tops=np.array([0.0]),
    )  # fmt: skip
    # 400 members: with 100, depth and fill resistivity can trade off and the whole
    # ensemble settle on a contact twice as deep that fits the data as well
    settings = bedrock.Settings(n_ensemble=400)
    out = bedrock.invert_line(ld, bedrock.make_prior(STATS, settings), settings, seed=0)
    assert np.all(out["depth_q10"] <= depth) and np.all(out["depth_q90"] >= depth)
    assert np.all(np.abs(out["depth_q50"] / depth - 1) < 0.25)
