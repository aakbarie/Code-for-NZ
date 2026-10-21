"""Linear-Gaussian state-space model: Kalman filter, RTS smoother, and EM.

    x_t = A x_{t-1} + w_t,          w_t ~ N(0, Q)        hidden state (k numbers)
    y_t = C x_t + D u_t + v_t,      v_t ~ N(0, diag R)   observation (p numbers)
    x_1 ~ N(mu0, P0)

u_t are known inputs (here: a constant and powers of log bird height), so D
absorbs whatever the inputs explain and the state is left with the rest.

Several independent sequences (flight lines) share one set of parameters.
Observations can be marked missing (e.g. the bird flew too high to see the
ground): the filter skips its update there, the smoother carries the state
through the gap with growing uncertainty, and EM leaves those rows out of the
observation-model updates.
Fitting is the EM algorithm of Shumway & Stoffer (1982): the E-step runs the
Kalman filter and Rauch-Tung-Striebel smoother; the M-step re-estimates A, C,
D, Q, R in closed form from the smoothed moments. Each EM step can only
increase the likelihood, which the tests check.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateSpace:
    A: np.ndarray  # (k, k) how the state carries over from one sounding to the next
    C: np.ndarray  # (p, k) how the state shows up in the observations
    D: np.ndarray  # (p, m) effect of the inputs
    Q: np.ndarray  # (k, k) how much the state can change per step
    R: np.ndarray  # (p,)   observation noise variances
    mu0: np.ndarray = field(default=None)  # (k,)
    P0: np.ndarray = field(default=None)  # (k, k)

    def __post_init__(self):
        k = self.A.shape[0]
        if self.mu0 is None:
            self.mu0 = np.zeros(k)
        if self.P0 is None:
            self.P0 = 10.0 * np.eye(k)  # vague: lines are long, the start hardly matters


@dataclass
class Filtered:
    x_pred: np.ndarray  # (n, k) state predicted from the soundings before t
    P_pred: np.ndarray  # (n, k, k)
    x_filt: np.ndarray  # (n, k) after also seeing sounding t
    P_filt: np.ndarray  # (n, k, k)
    innovation: np.ndarray  # (n, p) y_t minus its prediction
    nis: np.ndarray  # (n,) normalised innovation squared, ~ chi2(p) if the model holds
    loglik: float


def kalman_filter(
    m: StateSpace, Y: np.ndarray, U: np.ndarray, observed: np.ndarray | None = None
) -> Filtered:
    """Forward pass. observed[t] = False treats sounding t as missing."""
    n, p = Y.shape
    if observed is None:
        observed = np.ones(n, dtype=bool)
    k = m.A.shape[0]
    x_pred, x_filt = np.empty((n, k)), np.empty((n, k))
    P_pred, P_filt = np.empty((n, k, k)), np.empty((n, k, k))
    innovation, nis = np.empty((n, p)), np.empty(n)
    R = np.diag(m.R)
    obs_offset = U @ m.D.T
    x, P, loglik = m.mu0, m.P0, 0.0
    for t in range(n):
        if t:
            x = m.A @ x
            P = m.A @ P @ m.A.T + m.Q
        x_pred[t], P_pred[t] = x, P
        if not observed[t]:
            x_filt[t], P_filt[t] = x, P
            innovation[t], nis[t] = np.nan, np.nan
            continue
        e = Y[t] - m.C @ x - obs_offset[t]
        PCt = P @ m.C.T
        S = m.C @ PCt + R
        S_inv_e = np.linalg.solve(S, e)
        K = np.linalg.solve(S, PCt.T).T  # gain = P C' S^-1
        x = x + K @ e
        P = P - K @ PCt.T
        P = (P + P.T) / 2
        x_filt[t], P_filt[t] = x, P
        innovation[t], nis[t] = e, e @ S_inv_e
        loglik -= 0.5 * (np.linalg.slogdet(S)[1] + nis[t] + p * np.log(2 * np.pi))
    return Filtered(x_pred, P_pred, x_filt, P_filt, innovation, nis, loglik)


def rts_smoother(m: StateSpace, f: Filtered) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed states, covariances, and lag-one covariances Cov(x_t, x_{t-1} | all data)."""
    n, k = f.x_filt.shape
    xs, Ps = f.x_filt.copy(), f.P_filt.copy()
    J = np.zeros((n, k, k))
    for t in range(n - 2, -1, -1):
        J[t] = np.linalg.solve(f.P_pred[t + 1], m.A @ f.P_filt[t]).T  # P_filt A' P_pred^-1
        xs[t] = f.x_filt[t] + J[t] @ (xs[t + 1] - f.x_pred[t + 1])
        Ps[t] = f.P_filt[t] + J[t] @ (Ps[t + 1] - f.P_pred[t + 1]) @ J[t].T
    lag1 = np.einsum("tij,tkj->tik", Ps[1:], J[:-1])  # P_{t|n} J_{t-1}'
    return xs, Ps, lag1


def initial_guess(Ys: list[np.ndarray], Us: list[np.ndarray], k: int) -> StateSpace:
    """Regress out the inputs, then take the first k principal components as the state.

    Pass only observed rows; gaps are ignored for this rough start.
    """
    Y, U = np.vstack(Ys), np.vstack(Us)
    D = np.linalg.lstsq(U, Y, rcond=None)[0].T
    resid = Y - U @ D.T
    vals, vecs = np.linalg.eigh(np.cov(resid.T))
    order = np.argsort(vals)[::-1]
    C = vecs[:, order[:k]] * np.sqrt(vals[order[:k]])
    scores = resid @ vecs[:, order[:k]] / np.sqrt(vals[order[:k]])
    steps = np.vstack(
        [np.diff(s, axis=0) for s in np.split(scores, np.cumsum([len(y) for y in Ys])[:-1])]
    )
    R = np.maximum(np.var(resid - scores @ C.T, axis=0), 1e-4)
    return StateSpace(A=0.99 * np.eye(k), C=C, D=D, Q=np.cov(steps.T).reshape(k, k), R=R)


def em(
    Ys: list[np.ndarray],
    Us: list[np.ndarray],
    k: int,
    observed: list[np.ndarray] | None = None,
    max_iter: int = 50,
    tol: float = 1e-5,
    verbose: bool = False,
) -> tuple[StateSpace, list[float]]:
    """Fit by EM over several sequences. Returns the model and the log-likelihood history."""
    if observed is None:
        observed = [np.ones(len(Y), dtype=bool) for Y in Ys]
    m = initial_guess(
        [Y[o] for Y, o in zip(Ys, observed, strict=True)],
        [U[o] for U, o in zip(Us, observed, strict=True)],
        k,
    )
    p, q = Ys[0].shape[1], Us[0].shape[1]
    history: list[float] = []
    for it in range(max_iter):
        # E-step: accumulate smoothed moments over all sequences
        Sxx_obs = np.zeros((k, k))  # sum E[x_t x_t'], observed t
        Sxx_cur = np.zeros((k, k))  # sum E[x_t x_t'], t >= 2
        Sxx_prev = np.zeros((k, k))  # sum E[x_{t-1} x_{t-1}'], t >= 2
        Sx_lag = np.zeros((k, k))  # sum E[x_t x_{t-1}'], t >= 2
        Syx, Syu, Sxu = np.zeros((p, k)), np.zeros((p, q)), np.zeros((k, q))
        Suu, Syy = np.zeros((q, q)), np.zeros(p)
        n_obs = n_trans = 0
        loglik = 0.0
        for Y, U, o in zip(Ys, Us, observed, strict=True):
            f = kalman_filter(m, Y, U, o)
            xs, Ps, lag1 = rts_smoother(m, f)
            loglik += f.loglik
            Exx = Ps + np.einsum("ti,tj->tij", xs, xs)
            # state dynamics: every step
            Sxx_cur += Exx[1:].sum(axis=0)
            Sxx_prev += Exx[:-1].sum(axis=0)
            Sx_lag += (lag1 + np.einsum("ti,tj->tij", xs[1:], xs[:-1])).sum(axis=0)
            n_trans += len(Y) - 1
            # observation model: observed soundings only
            Sxx_obs += Exx[o].sum(axis=0)
            Syx += Y[o].T @ xs[o]
            Syu += Y[o].T @ U[o]
            Sxu += xs[o].T @ U[o]
            Suu += U[o].T @ U[o]
            Syy += (Y[o] ** 2).sum(axis=0)
            n_obs += int(o.sum())
        history.append(loglik)
        if verbose:
            print(f"  EM {it + 1:>2}: log-likelihood {loglik:,.1f}")
        if it and abs(history[-1] - history[-2]) < tol * abs(history[-2]):
            break

        # M-step: closed-form updates
        A = Sx_lag @ np.linalg.inv(Sxx_prev)
        Q = (Sxx_cur - A @ Sx_lag.T) / n_trans
        Ezz = np.block([[Sxx_obs, Sxu], [Sxu.T, Suu]])
        Eyz = np.hstack([Syx, Syu])
        CD = Eyz @ np.linalg.inv(Ezz)
        R = np.maximum((Syy - np.einsum("ij,ij->i", CD, Eyz)) / n_obs, 1e-6)
        m = StateSpace(A=A, C=CD[:, :k], D=CD[:, k:], Q=(Q + Q.T) / 2, R=R)
    return m, history


def canonical_rotation(m: StateSpace, xs: np.ndarray) -> np.ndarray:
    """Matrix T so that z = T (x - mean) has unit variance and orthogonal loadings.

    A state-space model is only defined up to a change of basis (C x = C T^-1 T x).
    This picks the basis where each feature's loading column is orthogonal,
    ordered by how much of the observations it explains, with a fixed sign.
    """
    L = np.linalg.cholesky(np.cov(xs.T).reshape(xs.shape[1], xs.shape[1]))
    _, s, Vt = np.linalg.svd(m.C @ L, full_matrices=False)
    T = Vt @ np.linalg.inv(L)
    loadings = m.C @ np.linalg.inv(T)
    signs = np.sign(loadings[np.abs(loadings).argmax(axis=0), range(len(s))])
    return signs[:, None] * T
