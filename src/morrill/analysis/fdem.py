"""1D forward model: what a helicopter EM bird measures over a layered earth.

A transmitter coil carries an alternating current, so it is a magnetic dipole
of moment m. Its field induces currents in the ground; their field (the
secondary field Hs) reaches the receiver coil a distance r away, together with
the transmitter's own field (the primary, Hp). The system reports Hs / Hp in
parts per million, in phase with the transmitter current (in-phase, I) and a
quarter cycle behind it (quadrature, Q).

Over horizontal layers the secondary field is a Hankel integral over the
horizontal wavenumber λ (Ward & Hohmann, 1988, eqs 4.46 and 4.105):

    coplanar (both coils vertical axis):
        Hs/Hp = -r³ ∫ r_TE(λ) e^{-2λh} λ² J0(λr) dλ
    coaxial (both coils' axes along the line between them):
        Hs/Hp = -(r³/2) ∫ r_TE(λ) e^{-2λh} λ² [J0(λr) - J1(λr)/(λr)] dλ

h is the bird's height and r_TE the earth's reflection coefficient, which
holds all the geology. It comes from a recursion up from the bottom layer
(reflection_te). For a perfect conductor r_TE = -1 and the integrals reduce to
the field of an image dipole a distance 2h below the bird, which the tests use.

The physical coaxial secondary field points against the primary. Fugro (and
the hemSystem.txt files, which give the coaxial transmitter a moment of -1)
report it with the sign flipped, so both orientations read positive over a
conductor; the formula above already includes that flip.

The integral is evaluated with the trapezoid rule in ln λ. The integrand
vanishes like λ³ at small λ and is cut off by e^{-2λh} at large λ, which makes
the rule converge very fast: 6 points per decade agree with adaptive
quadrature to within 0.01 ppm (the tests check this). Everything is
vectorised over models, so an ensemble of 100 earths costs one call.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
from scipy import special

MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class System:
    freq_hz: np.ndarray  # (F,)
    coil_sep_m: np.ndarray  # (F,)
    coaxial: np.ndarray  # (F,) bool
    noise_ppm: np.ndarray  # (F,) additive noise floor, same for I and Q

    @classmethod
    def from_db(cls, con: duckdb.DuckDBPyConnection) -> "System":
        rows = con.sql(
            "SELECT freq_hz, coil_sep_m, coil_orientation = 'coaxial', noise_ppm "
            "FROM hem_channel ORDER BY freq_hz"
        ).fetchall()
        f, r, cx, noise = map(np.array, zip(*rows, strict=True))
        return cls(f.astype(float), r.astype(float), cx.astype(bool), noise.astype(float))

    @property
    def channel_noise(self) -> np.ndarray:
        """Noise floor per channel, in the order of channels()."""
        return np.repeat(self.noise_ppm, 2)


@dataclass(frozen=True)
class Grid:
    lam: np.ndarray  # (L,) wavenumbers (1/m)
    weights: np.ndarray  # (L,) trapezoid weights for ∫ ... dλ


def hankel_grid(per_decade: int = 6, lo: float = 1e-4, hi: float = 1.0) -> Grid:
    """Equally spaced in ln λ. Fine for bird heights of about 10-150 m."""
    u = np.arange(np.log(lo), np.log(hi), np.log(10) / per_decade)
    lam = np.exp(u)
    return Grid(lam, (u[1] - u[0]) * lam)  # dλ = λ d(ln λ)


def reflection_te(
    lam: np.ndarray, omega: np.ndarray, sigma: np.ndarray, thickness: np.ndarray
) -> np.ndarray:
    """TE reflection coefficient of a layered earth: shape (models, frequencies, wavenumbers).

    sigma (M, N): conductivity (S/m) of N layers, the last a half-space;
    thickness (N-1,): metres. Quasi-static, all layers non-magnetic. Each
    layer j has a vertical wavenumber u_j = sqrt(λ² + iωμ0σ_j); û is the
    "apparent" u seen from the top of a layer, built from the bottom up.
    """
    u = np.sqrt(
        lam[None, None, :, None] ** 2
        + 1j * omega[None, :, None, None] * MU0 * sigma[:, None, None, :]
    )  # (M, F, L, N)
    u_hat = u[..., -1]
    for j in range(sigma.shape[1] - 2, -1, -1):
        uj = u[..., j]
        e = np.exp(-2 * uj * thickness[j])
        tanh = (1 - e) / (1 + e)  # tanh(u_j t_j), stable for any thickness
        u_hat = uj * (u_hat + uj * tanh) / (uj + u_hat * tanh)
    return (lam - u_hat) / (lam + u_hat)


def response(
    system: System,
    sigma: np.ndarray,
    thickness: np.ndarray,
    height_m: np.ndarray,
    grid: Grid | None = None,
) -> np.ndarray:
    """Hs/Hp in ppm, complex (real = in-phase, imaginary = quadrature): shape (M, F)."""
    grid = grid or hankel_grid()
    sigma = np.atleast_2d(sigma)
    height_m = np.broadcast_to(np.asarray(height_m, dtype=float), (sigma.shape[0],))
    lam, r = grid.lam, system.coil_sep_m[:, None]
    rte = reflection_te(lam, 2 * np.pi * system.freq_hz, sigma, np.asarray(thickness))
    lr = lam[None, :] * r  # (F, L)
    coplanar = r**3 * lam**2 * special.j0(lr)
    coaxial = 0.5 * r**3 * lam**2 * (special.j0(lr) - special.j1(lr) / lr)
    kernel = np.where(system.coaxial[:, None], coaxial, coplanar) * grid.weights  # (F, L)
    decay = np.exp(-2 * lam[None, :] * height_m[:, None])  # (M, L)
    return -1e6 * np.einsum("mfl,fl,ml->mf", rte, kernel, decay)


def channels(resp: np.ndarray) -> np.ndarray:
    """(M, F) complex -> (M, 2F) real: I and Q per frequency, as in hem_obs."""
    out = np.empty((resp.shape[0], 2 * resp.shape[1]))
    out[:, 0::2], out[:, 1::2] = resp.real, resp.imag
    return out
