import numpy as np
import pytest

from morrill.analysis import statespace as ss


def simulate(n_seq=3, n=400, seed=0):
    """Two smooth hidden states seen through 6 noisy channels plus a known input."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.0], [0.0, 0.95]])
    C = rng.normal(size=(6, 2))
    D = rng.normal(size=(6, 2))
    Ys, Us, Xs = [], [], []
    for _ in range(n_seq):
        x = np.zeros(2)
        X, Y = [], []
        U = np.c_[np.ones(n), np.sin(np.linspace(0, 6, n))]
        for t in range(n):
            x = A @ x + rng.normal(scale=0.2, size=2)
            X.append(x)
            Y.append(C @ x + D @ U[t] + rng.normal(scale=0.3, size=6))
        Xs.append(np.array(X))
        Ys.append(np.array(Y))
        Us.append(U)
    return Ys, Us, Xs, ss.StateSpace(A=A, C=C, D=D, Q=0.04 * np.eye(2), R=0.09 * np.ones(6))


def test_em_never_decreases_the_likelihood():
    Ys, Us, _, _ = simulate()
    _, history = ss.em(Ys, Us, k=2, max_iter=15, tol=0)
    assert np.all(np.diff(history) > -1e-6 * np.abs(history[:-1]))


def test_smoother_beats_filter_on_the_true_model():
    Ys, Us, Xs, truth = simulate(n_seq=1)
    f = ss.kalman_filter(truth, Ys[0], Us[0])
    xs, _, _ = ss.rts_smoother(truth, f)
    err_filter = np.mean((f.x_filt - Xs[0]) ** 2)
    err_smooth = np.mean((xs - Xs[0]) ** 2)
    assert err_smooth < err_filter


def test_fitted_model_recovers_the_hidden_signal():
    Ys, Us, Xs, _ = simulate()
    m, _ = ss.em(Ys, Us, k=2)
    f = ss.kalman_filter(m, Ys[0], Us[0])
    xs, _, _ = ss.rts_smoother(m, f)
    # the states are only defined up to a change of basis, so regress truth on estimate
    coef, *_ = np.linalg.lstsq(np.c_[xs, np.ones(len(xs))], Xs[0], rcond=None)
    fit = np.c_[xs, np.ones(len(xs))] @ coef
    r2 = 1 - ((Xs[0] - fit) ** 2).sum(axis=0) / ((Xs[0] - Xs[0].mean(axis=0)) ** 2).sum(axis=0)
    assert np.all(r2 > 0.8)
    # and the input effect D is recovered without being absorbed into the state
    assert m.D[:, 1] == pytest.approx(simulate()[3].D[:, 1], abs=0.3)


def test_innovations_flag_a_spike():
    Ys, Us, _, truth = simulate(n_seq=1)
    Y = Ys[0].copy()
    Y[200] += 5.0
    f = ss.kalman_filter(truth, Y, Us[0])
    assert f.nis.argmax() == 200


def test_canonical_rotation_whitens_the_states():
    Ys, Us, _, _ = simulate()
    m, _ = ss.em(Ys, Us, k=2)
    xs, _, _ = ss.rts_smoother(m, ss.kalman_filter(m, Ys[0], Us[0]))
    T = ss.canonical_rotation(m, xs)
    z = (xs - xs.mean(axis=0)) @ T.T
    assert np.cov(z.T) == pytest.approx(np.eye(2), abs=1e-8)


def test_missing_stretch_is_bridged_with_growing_uncertainty():
    Ys, Us, Xs, truth = simulate(n_seq=1)
    observed = np.ones(len(Ys[0]), dtype=bool)
    observed[150:200] = False
    f = ss.kalman_filter(truth, Ys[0], Us[0], observed)
    xs, Ps, _ = ss.rts_smoother(truth, f)
    assert np.isnan(f.nis[150:200]).all() and not np.isnan(f.nis[:150]).any()
    var = Ps[:, 0, 0]
    assert var[175] > 3 * var[100]  # most uncertain in the middle of the gap
    # the smoother still tracks the truth through the gap better than ignoring it would
    assert np.mean((xs[150:200] - Xs[0][150:200]) ** 2) < np.mean(Xs[0][150:200] ** 2)


def test_em_with_missing_rows_never_decreases_the_likelihood():
    Ys, Us, _, _ = simulate()
    observed = [np.r_[np.ones(300, bool), np.zeros(50, bool), np.ones(50, bool)] for _ in Ys]
    _, history = ss.em(Ys, Us, k=2, observed=observed, max_iter=15, tol=0)
    assert np.all(np.diff(history) > -1e-6 * np.abs(history[:-1]))
