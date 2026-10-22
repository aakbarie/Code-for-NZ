"""Inversion as a profunctor: compatibilities between two spaces, composed through a third.

A forward model is a function: one earth, one set of data. An inversion is not,
because many earths fit the same data. The right shape is a profunctor. Here
that is an array of compatibilities between two grids, together with a rule for
composing two such arrays through the grid they share:

    (Q ∘ P)[a, c] = ⊕_b  P[a, b] ⊗ Q[b, c]

The rule is a semiring (⊕, ⊗). Swapping it changes the question while the code
stays the same:

- SUM_PRODUCT, in logs (⊕ = log-sum-exp, ⊗ = +): probabilities. Composing along
  a flight line is the forward-backward algorithm, of which the Kalman filter
  and RTS smoother (statespace.py, enki.py) are the Gaussian special case.
- MAX_PLUS (⊕ = max, ⊗ = +): log-probabilities read as negative costs.
  Composing along the line is dynamic programming (Viterbi): the single most
  probable sequence.

Everything lives on a grid, so nothing is assumed Gaussian, and a posterior
with two separate answers keeps both.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class Semiring:
    name: str
    add: Callable[..., np.ndarray]  # ⊕, reducing one axis; ⊗ is + throughout (we work in logs)


SUM_PRODUCT = Semiring("sum-product", lambda x, axis: logsumexp(x, axis=axis))
MAX_PLUS = Semiring("max-plus", lambda x, axis: np.max(x, axis=axis))


def compose(P: np.ndarray, Q: np.ndarray, semiring: Semiring = SUM_PRODUCT) -> np.ndarray:
    """(Q ∘ P)[a, c] = ⊕_b P[a, b] + Q[b, c], for log-arrays P (A, B) and Q (B, C)."""
    return semiring.add(P[:, :, None] + Q[None, :, :], axis=1)


def chain(log_prior, log_kernels, log_liks, semiring: Semiring = SUM_PRODUCT):
    """Compose a line of n steps on a grid of k states.

    log_prior (k,): the first state; log_kernels (n-1, k, k): step t to t+1;
    log_liks (n, k): each step's compatibility with its data.
    Returns forward and backward messages (n, k) and the log evidence. With
    SUM_PRODUCT, fwd + bwd - evidence is the log posterior at each step.
    """
    n, k = log_liks.shape
    fwd, bwd = np.empty((n, k)), np.zeros((n, k))
    fwd[0] = log_prior + log_liks[0]
    for t in range(1, n):
        fwd[t] = semiring.add(fwd[t - 1][:, None] + log_kernels[t - 1], axis=0) + log_liks[t]
    for t in range(n - 2, -1, -1):
        bwd[t] = semiring.add(log_kernels[t] + (log_liks[t + 1] + bwd[t + 1])[None, :], axis=1)
    return fwd, bwd, float(semiring.add(fwd[-1], axis=0))


class Line:
    """A line's kernels held in linear space, for fast SUM_PRODUCT composition: one
    k×k matrix-vector product per step, rescaled at every step so nothing underflows.
    Same answers as chain(..., SUM_PRODUCT), many times faster in a loop."""

    def __init__(self, log_prior, log_kernels):
        self.log_scale = float(log_prior.max())
        self.prior = np.exp(log_prior - self.log_scale)
        self.T = np.exp(log_kernels)

    def compose(self, log_liks, backward: bool = True):
        """Scaled forward (and backward) messages, and the log evidence."""
        n, k = log_liks.shape
        top = log_liks.max(axis=1, keepdims=True)
        lik = np.exp(log_liks - top)
        fwd = np.empty((n, k))
        c = np.empty(n)
        a = self.prior * lik[0]
        c[0] = a.sum()
        fwd[0] = a / c[0]
        for t in range(1, n):
            a = (fwd[t - 1] @ self.T[t - 1]) * lik[t]
            c[t] = a.sum()
            fwd[t] = a / c[t]
        evidence = float(np.log(c).sum() + top.sum() + self.log_scale)
        if not backward:
            return fwd, None, evidence
        bwd = np.empty((n, k))
        bwd[-1] = 1.0
        for t in range(n - 2, -1, -1):
            b = self.T[t] @ (lik[t + 1] * bwd[t + 1])
            bwd[t] = b / b.sum()
        return fwd, bwd, evidence

    @staticmethod
    def marginals(fwd, bwd) -> np.ndarray:
        p = fwd * bwd
        return p / p.sum(axis=1, keepdims=True)

    def sample(self, fwd, rng) -> np.ndarray:
        """One exact joint draw of the whole sequence (backward sampling)."""
        n, k = fwd.shape
        out = np.empty(n, dtype=int)
        u = rng.random(n)
        out[-1] = np.searchsorted(np.cumsum(fwd[-1]), u[-1] * fwd[-1].sum())
        for t in range(n - 2, -1, -1):
            w = np.cumsum(fwd[t] * self.T[t][:, out[t + 1]])
            out[t] = np.searchsorted(w, u[t] * w[-1])
        return np.minimum(out, k - 1)


def posterior(log_prior, log_kernels, log_liks) -> tuple[np.ndarray, float]:
    """Log posterior at every step (n, k), and the log evidence."""
    fwd, bwd, evidence = chain(log_prior, log_kernels, log_liks, SUM_PRODUCT)
    return fwd + bwd - evidence, evidence


def best_path(log_prior, log_kernels, log_liks) -> np.ndarray:
    """MAX_PLUS composition with back-pointers: indices of the most probable sequence."""
    n, k = log_liks.shape
    score = log_prior + log_liks[0]
    back = np.empty((n, k), dtype=int)
    for t in range(1, n):
        cand = score[:, None] + log_kernels[t - 1]
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + log_liks[t]
    path = np.empty(n, dtype=int)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def sample_paths(fwd, log_kernels, rng, n_samples: int) -> np.ndarray:
    """Exact joint draws of whole sequences (forward filtering, backward sampling): (n, N)."""
    n, k = fwd.shape

    def draw(logw):  # one categorical draw per row of logw (N, k)
        p = np.exp(logw - logw.max(axis=1, keepdims=True))
        c = np.cumsum(p, axis=1)
        return (c < rng.random((len(c), 1)) * c[:, -1:]).sum(axis=1)

    out = np.empty((n, n_samples), dtype=int)
    out[-1] = draw(np.tile(fwd[-1], (n_samples, 1)))
    for t in range(n - 2, -1, -1):
        out[t] = draw(fwd[t][None, :] + log_kernels[t][:, out[t + 1]].T)
    return out


def reversible_kernel(x, log_prior, mean: float, sd: float, a: float) -> np.ndarray:
    """An AR(1) step with correlation a on grid x, as a log-kernel (k, k) that leaves
    log_prior exactly invariant: the discretized Gaussian conditional, corrected by
    Metropolis-Hastings (on a fine grid it accepts nearly everything)."""
    cond_mean = mean + a * (x - mean)
    cond_sd = max(sd * np.sqrt(max(1 - a**2, 0.0)), 1e-9)
    log_q = -0.5 * ((x[None, :] - cond_mean[:, None]) / cond_sd) ** 2
    log_q -= logsumexp(log_q, axis=1, keepdims=True)
    log_accept = np.minimum(0.0, (log_prior[None, :] + log_q.T) - (log_prior[:, None] + log_q))
    T = np.exp(log_q + log_accept)
    np.fill_diagonal(T, 0.0)
    T[np.diag_indices_from(T)] = np.maximum(1.0 - T.sum(axis=1), 0.0)
    with np.errstate(divide="ignore"):
        return np.log(T)


def grid_log_prior(x, mean: float, sd: float) -> np.ndarray:
    """A Gaussian prior on an evenly spaced grid, normalized to sum to one."""
    lp = -0.5 * ((x - mean) / sd) ** 2
    return lp - logsumexp(lp)


def quantiles(log_post, x, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
    """Quantiles (len(qs), n) of each step's posterior, reading each grid node as a cell
    of the evenly spaced grid x and interpolating the CDF linearly inside it."""
    p = np.exp(log_post - logsumexp(log_post, axis=1, keepdims=True))
    dx = x[1] - x[0]
    edges = np.r_[x - dx / 2, x[-1] + dx / 2]
    cdf = np.c_[np.zeros(len(p)), np.cumsum(p, axis=1)]
    return np.array([[np.interp(q, c, edges) for c in cdf] for q in qs])


def n_modes(log_post, min_mass: float = 0.1) -> np.ndarray:
    """How many separate answers each step's posterior holds: stretches between local
    minima of the (lightly smoothed) posterior that each carry at least min_mass."""
    p = np.exp(log_post - logsumexp(log_post, axis=1, keepdims=True))
    out = np.empty(len(p), dtype=int)
    for t, row in enumerate(p):
        s = np.convolve(row, [0.25, 0.5, 0.25], mode="same")
        minima = np.flatnonzero((s[1:-1] < s[:-2]) & (s[1:-1] <= s[2:])) + 1
        mass = np.add.reduceat(row, np.r_[0, minima])
        out[t] = int((mass >= min_mass).sum())
    return out
