import numpy as np
import pytest
from scipy import integrate, special

from morrill.analysis import fdem

SYSTEM = fdem.System(
    freq_hz=np.array([378.0, 1776.0, 3345.0, 8171.0, 41020.0, 129550.0]),
    coil_sep_m=np.array([7.93, 7.91, 9.03, 7.91, 7.91, 7.89]),
    coaxial=np.array([False, False, True, False, False, False]),
    noise_ppm=np.array([5.0, 10.0, 10.0, 10.0, 20.0, 40.0]),
)
THICK = np.array([2.0, 3.0, 5.0, 10.0, 20.0])


def image_ppm(r, h, coaxial):
    """A perfect conductor's response: the field of an image dipole 2h below the bird."""
    R = np.hypot(r, 2 * h)
    if coaxial:  # sign flipped, as reported
        return 1e6 * r**3 * (1 - 3 * r**2 / R**2) / (2 * R**3)
    return 1e6 * r**3 * (8 * h**2 - r**2) / R**5


@pytest.mark.parametrize("h", [15.0, 30.0, 60.0])
def test_perfect_conductor_matches_the_image_dipole(h):
    got = fdem.response(SYSTEM, np.array([[1e12]]), np.array([]), h)[0]
    for f in range(len(SYSTEM.freq_hz)):
        exact = image_ppm(SYSTEM.coil_sep_m[f], h, SYSTEM.coaxial[f])
        assert got[f].real == pytest.approx(exact, rel=1e-5)
        assert abs(got[f].imag) < 1e-3 * exact


def brute_force(sigma, thick, h, f):
    omega = np.array([2 * np.pi * SYSTEM.freq_hz[f]])
    r, cx = SYSTEM.coil_sep_m[f], SYSTEM.coaxial[f]

    def integrand(lam, part):
        rte = fdem.reflection_te(np.array([lam]), omega, sigma[None, :], thick)[0, 0, 0]
        lr = lam * r
        k = (
            0.5 * r**3 * lam**2 * (special.j0(lr) - special.j1(lr) / lr)
            if cx
            else r**3 * lam**2 * special.j0(lr)
        )
        v = -1e6 * rte * np.exp(-2 * lam * h) * k
        return v.real if part == "re" else v.imag

    re, im = (
        integrate.quad(integrand, 0, 60 / h, args=(p,), limit=500, epsabs=1e-7)[0]
        for p in ("re", "im")
    )
    return re + 1j * im


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_layered_earth_matches_adaptive_quadrature(seed):
    rng = np.random.default_rng(seed)
    sigma = 10 ** -rng.uniform(0, 3, size=len(THICK) + 1)
    h = rng.uniform(15, 60)
    got = fdem.response(SYSTEM, sigma[None, :], THICK, h)[0]
    for f in range(len(SYSTEM.freq_hz)):
        assert abs(got[f] - brute_force(sigma, THICK, h, f)) < 0.01  # ppm; noise is 5-40 ppm


def test_splitting_a_layer_changes_nothing():
    sigma = np.array([0.01, 0.1, 0.02])
    whole = fdem.response(SYSTEM, sigma[None, :], np.array([4.0, 6.0]), 30.0)
    split_sigma = np.array([[0.01, 0.01, 0.1, 0.1, 0.02]])
    split = fdem.response(SYSTEM, split_sigma, np.array([1.5, 2.5, 2.0, 4.0]), 30.0)
    np.testing.assert_allclose(split, whole, rtol=1e-10)


def test_more_conductive_ground_gives_a_bigger_response_and_positive_quadrature():
    sig = np.array([[0.002], [0.02], [0.2]])
    resp = fdem.response(SYSTEM, sig, np.array([]), 30.0)
    assert np.all(np.diff(resp.real, axis=0) > 0)
    assert np.all(resp.imag > 0)


def test_ensemble_and_single_calls_agree():
    rng = np.random.default_rng(3)
    sigma = 10 ** -rng.uniform(0, 3, size=(5, len(THICK) + 1))
    heights = rng.uniform(20, 50, 5)
    batch = fdem.response(SYSTEM, sigma, THICK, heights)
    for i in range(5):
        np.testing.assert_allclose(batch[i], fdem.response(SYSTEM, sigma[i], THICK, heights[i])[0])
    assert fdem.channels(batch).shape == (5, 12)
    assert fdem.channels(batch)[0, 1] == batch[0, 0].imag


def test_per_model_thicknesses_match_one_call_per_model():
    rng = np.random.default_rng(5)
    sigma = 10 ** -rng.uniform(0, 3, size=(4, 3))
    thick = rng.uniform(1, 30, size=(4, 2))
    batch = fdem.response(SYSTEM, sigma, thick, 30.0)
    for i in range(4):
        np.testing.assert_allclose(batch[i], fdem.response(SYSTEM, sigma[i], thick[i], 30.0)[0])
