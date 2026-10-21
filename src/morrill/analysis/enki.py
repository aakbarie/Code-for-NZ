"""Ensemble Kalman inversion (EnKI) of the HEM data, sounding by sounding along each line.

Each sounding's unknowns are
    θ = [log10 ρ of the 24 layers and the half-space used by EM1DFM, δh]
where δh corrects the laser altimeter's bird height. An ensemble of N earths
stands in for the probability distribution of θ; the physics is the forward
model in fdem.py.

Prior, learned from data rather than set by hand:
- log10 ρ: mean and spread of the borehole induction logs (averaged into the
  model layers), with their vertical correlation
      corr(Δz) = c + (1 - c) exp(-Δz / L_v),
  fitted to the logs. c is the share of variation that moves the whole
  profile (one hole is more resistive than another throughout); L_v is how
  far down a local change carries.
- δh ~ N(0, HEIGHT_SD²).
- Along the line, θ is an AR(1) process, the discrete form of a Gaussian
  process with exponential correlation exp(-distance / L_h):
      θ_t = m + a (θ_{t-1} - m) + w,   a = exp(-Δ / L_h),   w ~ N(0, (1 - a²) Σ).
  At every sounding its distribution is exactly the prior N(m, Σ); L_h only
  says how far along the line knowledge carries. L_h, the prior's width and
  c are chosen by maximum marginal likelihood (select_settings); the data
  error is not (see NOISE_GRID).

Each sounding (the "coupled" mode):
1. Forecast: push the previous sounding's ensemble one step along the line.
2. Analysis: ES-MDA (Emerick & Reynolds, 2013). The Kalman update
       θ_i += C_θd (C_dd + α R)⁻¹ (d_obs + √α ε_i - g(θ_i)),
   with covariances taken from the ensemble, applied N_a times with the data
   error inflated by α = N_a. For a linear g this reproduces the exact
   Gaussian posterior; for the nonlinear EM physics the repeated small steps
   follow the curvature.
Then an ensemble Rauch-Tung-Striebel smoother runs back along the line, so
each sounding's answer uses the data on both sides (as in note 05).

The "independent" mode starts every sounding from the same prior, so nothing
is borrowed from the neighbours: a per-sounding Bayesian inversion, the job
the MCMC did.

Soundings where the bird flew above MAX_HEIGHT_M are forecast but not updated.

Before any of this, each channel gets a survey-wide gain (calibrate): three
channels disagree with the rest by 10-15% at every sounding, which no 1D
earth can explain.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from morrill.analysis import fdem
from morrill.ingest.airborne import CHANNELS

CHANNEL_COLS = list(CHANNELS.values())
INVERTED_RUN = "em1dfm_pt1"
MAX_HEIGHT_M = 60.0
CONDUCTOR_OHM_M = 20.0  # same rule as the model_conductor_depth view
CONDUCTOR_MIN_TOP_M = 1.8
LOG_RHO_RANGE = (-0.5, 3.5)  # 0.3 to 3,000 ohm·m: members outside are clipped
HEIGHT_SD = 2.0  # m
HEIGHT_RANGE = (-10.0, 10.0)


@dataclass(frozen=True)
class Settings:
    n_ensemble: int = 400  # 100 members understate the spread by about a quarter
    n_assim: int = 4  # ES-MDA passes per sounding; 8 or 16 fit no better
    step: int = 20  # every 20th sounding, about 64 m: roughly one footprint apart
    # chosen by marginal likelihood (select_settings); these defaults are its answer
    length_m: float = 3000.0  # L_h, along-line correlation length
    # data error = sqrt(floor² + (rel_noise · |d|)²). Not chosen by likelihood: see NOISE_GRID
    rel_noise: float = 0.05
    prior_sd_scale: float = 1.5  # multiplies the logs' spread of log10 ρ
    whole_profile_share: float = 0.0  # c; the logs can't pin it down (see LogStats)


@dataclass(frozen=True)
class Prior:
    mean: np.ndarray  # (p,)
    cov: np.ndarray  # (p, p)
    info: dict

    @property
    def chol(self) -> np.ndarray:
        return np.linalg.cholesky(self.cov)


# --- layers and prior ---------------------------------------------------------------------


def layers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    lay = con.sql("SELECT layer, top_m, thickness_m FROM model_layer ORDER BY layer").df()
    # the half-space has no thickness; give its centre 10 m below its top for correlations
    lay["mid_m"] = lay["top_m"] + lay["thickness_m"].fillna(20.0) / 2
    return lay


def _vertical_corr(dz, c, length):
    return c + (1 - c) * np.exp(-dz / length)


@dataclass(frozen=True)
class LogStats:
    """What the induction logs say about log10 ρ in the model layers.

    The correlation between two layers' anomalies is measured across holes.
    With 6-19 holes per pair it is noisy (standard error about 0.25), so the
    logs fix the short-range shape well but not the whole-profile share c:
    the 13 holes near the survey suggest about 0.7, the 19 others about 0.
    """

    mid_m: np.ndarray  # layer centres
    mean: float
    sd: float
    dz: np.ndarray  # layer separations with a measured correlation
    rho: np.ndarray  # the correlations
    n: np.ndarray  # holes behind each
    holes: int

    def fit(self, c: float | None = None) -> tuple[float, float]:
        """(c, L_v) by weighted least squares; with c given, only L_v is fitted."""
        sigma = 1 / np.sqrt(self.n)
        if c is None:
            (c, length), _ = curve_fit(
                _vertical_corr, self.dz, self.rho, p0=(0.4, 5.0), sigma=sigma,
                bounds=([0, 0.5], [0.95, 100]),
            )  # fmt: skip
            return float(c), float(length)
        (length,), _ = curve_fit(
            lambda dz, L: _vertical_corr(dz, c, L), self.dz, self.rho, p0=(5.0,), sigma=sigma,
            bounds=([0.5], [100]),
        )  # fmt: skip
        return c, float(length)


def log_stats(
    con: duckdb.DuckDBPyConnection, exclude: tuple[str, ...] = (), min_coverage: float = 0.8
) -> LogStats:
    """Induction-log statistics in the model layers; holes in `exclude` left out."""
    lay = layers(con)
    logs = con.sql(
        "SELECT hole_id, layer, log10(resistivity_ohm_m) AS lr FROM borehole_layer_resistivity "
        f"WHERE coverage > {min_coverage}"
    ).df()
    logs = logs[~logs["hole_id"].isin(exclude)]
    wide = logs.pivot(index="hole_id", columns="layer", values="lr")
    anomaly = wide - wide.mean()
    mid = lay.set_index("layer")["mid_m"]
    dz, rho, n = [], [], []
    for i in wide.columns:
        for j in wide.columns:
            ok = anomaly[i].notna() & anomaly[j].notna()
            if j > i and ok.sum() >= 6:
                dz.append(abs(mid[j] - mid[i]))
                rho.append(np.corrcoef(anomaly[i][ok], anomaly[j][ok])[0, 1])
                n.append(ok.sum())
    return LogStats(
        lay["mid_m"].to_numpy(), float(logs["lr"].mean()), float(logs["lr"].std()),
        np.array(dz), np.array(rho), np.array(n), int(wide.shape[0]),
    )  # fmt: skip


def build_prior(stats: LogStats, settings: Settings) -> Prior:
    c, length = stats.fit(settings.whole_profile_share)
    sd = stats.sd * settings.prior_sd_scale
    info = {"mean_log10_rho": stats.mean, "sd_log10_rho": sd, "c": c, "vertical_length_m": length}
    return make_prior(stats.mid_m, stats.mean, sd, c, length, info)


def make_prior(mid_m, mean, sd, c, length_m, info=None) -> Prior:
    """N(m, Σ) on [log10 ρ per layer, δh] with the vertical correlation above."""
    n = len(mid_m)
    cov = np.zeros((n + 1, n + 1))
    cov[:n, :n] = sd**2 * _vertical_corr(np.abs(np.subtract.outer(mid_m, mid_m)), c, length_m)
    cov[n, n] = HEIGHT_SD**2
    return Prior(np.r_[np.full(n, mean), 0.0], cov, info or {})


# --- one sounding -------------------------------------------------------------------------


def data_error(d_obs: np.ndarray, floor: np.ndarray, rel_noise: float) -> np.ndarray:
    return np.sqrt(floor**2 + (rel_noise * d_obs) ** 2)


def clip(theta: np.ndarray) -> np.ndarray:
    theta[:, :-1] = np.clip(theta[:, :-1], *LOG_RHO_RANGE)
    theta[:, -1] = np.clip(theta[:, -1], *HEIGHT_RANGE)
    return theta


def gaussian_logpdf(y: np.ndarray, D: np.ndarray) -> float:
    """log N(y; mean of D, covariance of D + I): how well the forecast predicted the data."""
    r = y - D.mean(axis=0)
    C = np.cov(D.T) + np.eye(len(y))
    _, logdet = np.linalg.slogdet(C)
    return -0.5 * (r @ np.linalg.solve(C, r) + logdet + len(y) * np.log(2 * np.pi))


def es_mda(theta, y, forward, rng, n_assim, bounded=True):
    """Ensemble smoother with multiple data assimilation, in whitened data units (R = I).

    theta (N, p) prior ensemble; y (m,) data / error; forward(theta) -> (N, m) whitened.
    Returns the updated ensemble and the forecast's log predictive density of y.
    """
    n = theta.shape[0]
    alpha = float(n_assim)  # sum of 1/alpha over the passes is 1
    for k in range(n_assim):
        D = forward(theta)
        if k == 0:
            logpdf = gaussian_logpdf(y, D)
        dT = theta - theta.mean(axis=0)
        dD = D - D.mean(axis=0)
        C_td = dT.T @ dD / (n - 1)
        C_dd = dD.T @ dD / (n - 1)
        perturbed = y + np.sqrt(alpha) * rng.standard_normal(D.shape)
        gain = np.linalg.solve(C_dd + alpha * np.eye(len(y)), C_td.T).T  # (p, m)
        theta = theta + (perturbed - D) @ gain.T
        if bounded:
            theta = clip(theta)
    return theta, logpdf


# --- along a line -------------------------------------------------------------------------


def forecast(theta, prior: Prior, a: float, rng) -> np.ndarray:
    """One AR(1) step. If theta ~ N(m, Σ), so is the result."""
    noise = rng.standard_normal(theta.shape) @ prior.chol.T
    return prior.mean + a * (theta - prior.mean) + np.sqrt(1 - a * a) * noise


def filter_line(Y, forwards, step_m, observed, prior, settings, rng, bounded=True):
    """Forward pass. Y (n, m) whitened data; forwards[t](theta) whitened predictions at t.

    Returns forecast and analysis ensembles (n, N, p) and per-step log predictive density.
    """
    n, p = len(Y), len(prior.mean)
    F = np.empty((n, settings.n_ensemble, p))
    A = np.empty_like(F)
    logpdf = np.full(n, np.nan)
    theta = prior.mean + rng.standard_normal((settings.n_ensemble, p)) @ prior.chol.T
    for t in range(n):
        if t:
            theta = forecast(theta, prior, np.exp(-step_m[t] / settings.length_m), rng)
        F[t] = theta
        if observed[t]:
            theta, logpdf[t] = es_mda(theta, Y[t], forwards[t], rng, settings.n_assim, bounded)
        A[t] = theta
    return F, A, logpdf


def rts_smoother(F: np.ndarray, A: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    """Ensemble RTS smoother (Raanes, 2016): S_t = A_t + (S_{t+1} - F_{t+1}) J_t'."""
    S = A.copy()
    n_ens, p = A.shape[1:]
    for t in range(len(A) - 2, -1, -1):
        dA = A[t] - A[t].mean(axis=0)
        dF = F[t + 1] - F[t + 1].mean(axis=0)
        C_af = dA.T @ dF / (n_ens - 1)
        C_ff = dF.T @ dF / (n_ens - 1)
        J = np.linalg.solve(C_ff + ridge * np.trace(C_ff) / p * np.eye(p), C_af.T).T
        S[t] = A[t] + (S[t + 1] - F[t + 1]) @ J.T
    return S


# --- summaries ----------------------------------------------------------------------------


def conductor_depth(log_rho: np.ndarray, tops: np.ndarray) -> np.ndarray:
    """Top of the first layer at or below 1.8 m with ρ < 20 ohm·m; inf if there is none."""
    hit = (log_rho < np.log10(CONDUCTOR_OHM_M)) & (tops >= CONDUCTOR_MIN_TOP_M)
    return np.where(hit.any(axis=-1), tops[hit.argmax(axis=-1)], np.inf)


def summarise(ens: np.ndarray, tops: np.ndarray) -> dict[str, np.ndarray]:
    """Per step: quantiles of log10 ρ per layer, of conductor depth, and of δh."""
    q = np.quantile(ens[..., :-1], [0.1, 0.5, 0.9], axis=1)  # (3, n, layers)
    depth = conductor_depth(ens[..., :-1], tops)  # (n, N)
    dq = np.quantile(depth, [0.1, 0.5, 0.9], axis=1, method="inverted_cdf")
    return {
        "log_rho_q10": q[0], "log_rho_q50": q[1], "log_rho_q90": q[2],
        "depth_q10": dq[0], "depth_q50": dq[1], "depth_q90": dq[2],
        "p_no_conductor": np.isinf(depth).mean(axis=1),
        "dh_q50": np.median(ens[..., -1], axis=1),
    }  # fmt: skip


# --- running a line -----------------------------------------------------------------------


@dataclass
class LineData:
    line: int
    sounding_id: np.ndarray  # (n,) the soundings used (every `step`-th)
    xy: np.ndarray  # (n, 2)
    d_obs: np.ndarray  # (n, 12) ppm
    height_m: np.ndarray  # (n,)
    step_m: np.ndarray  # (n,) distance from the previous step
    system: fdem.System
    thickness: np.ndarray
    tops: np.ndarray

    def subset(self, idx: np.ndarray) -> "LineData":
        """Selected steps only: fine for the independent mode, which ignores neighbours."""
        return replace(
            self, sounding_id=self.sounding_id[idx], xy=self.xy[idx], d_obs=self.d_obs[idx],
            height_m=self.height_m[idx], step_m=np.full(len(idx), np.nan),
        )  # fmt: skip


def line_data(survey: pd.DataFrame, line: int, step: int, system, lay) -> LineData:
    sel = survey[survey["line"] == line].iloc[::step]
    xy = sel[["x", "y"]].to_numpy()
    return LineData(
        line, sel["sounding_id"].to_numpy(), xy, sel[CHANNEL_COLS].to_numpy(),
        sel["bird_height_m"].to_numpy(), np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)], system,
        lay["thickness_m"].dropna().to_numpy(), lay["top_m"].to_numpy(),
    )  # fmt: skip


def _forwards(ld: LineData, sd: np.ndarray):
    def make(t):
        def g(theta):
            sigma = 10.0 ** -theta[:, :-1]
            h = np.maximum(ld.height_m[t] + theta[:, -1], 5.0)
            return fdem.channels(fdem.response(ld.system, sigma, ld.thickness, h)) / sd[t]

        return g

    return [make(t) for t in range(len(ld.d_obs))]


def _residuals(ens: np.ndarray, Y: np.ndarray, forwards, observed) -> np.ndarray:
    """Whitened residual of the ensemble-mean model at each step, (n, 12); NaN where unobserved."""
    mean = ens.mean(axis=1)
    res = np.full(Y.shape, np.nan)
    for t in np.flatnonzero(observed):
        res[t] = forwards[t](mean[t : t + 1])[0] - Y[t]
    return res


def invert_line(ld: LineData, prior: Prior, settings: Settings, seed: int, modes=("coupled",)):
    """{mode: per-step summaries}; the coupled run adds its log marginal likelihood."""
    rng = np.random.default_rng(seed)
    sd = data_error(ld.d_obs, ld.system.channel_noise, settings.rel_noise)
    Y = ld.d_obs / sd
    observed = ld.height_m <= MAX_HEIGHT_M
    forwards = _forwards(ld, sd)
    out = {}
    if "coupled" in modes:
        F, A, logpdf = filter_line(Y, forwards, ld.step_m, observed, prior, settings, rng)
        S = rts_smoother(F, A)
        out["coupled"] = {**summarise(S, ld.tops), "residual": _residuals(S, Y, forwards, observed)}
        # log p(data | settings) in ppm units: whitening the data adds -sum(log sd)
        out["loglik"] = np.nansum(logpdf) - np.log(sd[observed]).sum()
    if "independent" in modes:
        ens = np.empty((len(Y), settings.n_ensemble, len(prior.mean)))
        for t in range(len(Y)):
            theta = prior.mean + rng.standard_normal(ens.shape[1:]) @ prior.chol.T
            if observed[t]:
                theta, _ = es_mda(theta, Y[t], forwards[t], rng, settings.n_assim)
            ens[t] = theta
        out["independent"] = {
            **summarise(ens, ld.tops),
            "residual": _residuals(ens, Y, forwards, observed),
        }
    return out


def _invert_job(args):
    ld, prior, settings, seed, modes = args
    return invert_line(ld, prior, settings, seed, modes)


def invert_lines(lines_data, prior, settings, modes=("coupled",), workers=4, seed=0):
    """Invert several lines in parallel: {line: invert_line output}."""
    jobs = [(ld, prior, settings, seed + ld.line, modes) for ld in lines_data]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(zip([ld.line for ld in lines_data], pool.map(_invert_job, jobs), strict=True))


def calibrate(ld: LineData, stats: LogStats, settings: Settings, n_iter: int = 1):
    """One gain per channel for the whole survey, estimated jointly with the geology.

    Alternate: invert with the current gains, then set each gain to the median
    of measured / predicted. A constant factor on one channel at every
    sounding can't be geology, which changes from sounding to sounding. But a
    factor common to all channels looks like the bird flying a little lower,
    which δh absorbs; so the gains are only defined relative to each other,
    and their geometric mean is held at 1.

    Iterating does not converge for every channel. The coaxial pair and the
    8.2 kHz in-phase settle in the first pass and stay; the 400 Hz in-phase
    and 140 kHz gains keep drifting, because deep conductivity and bird height
    can trade off against them. So the default is one pass: gains from a
    model fitted to the uncorrected data.
    Returns the gains (measured = gain x true) and their history.
    """
    gains, history = np.ones(ld.d_obs.shape[1]), []
    prior = build_prior(stats, settings)
    for _ in range(n_iter):
        fixed = replace(ld, d_obs=ld.d_obs / gains)
        res = invert_line(fixed, prior, settings, seed=1)["coupled"]["residual"]
        sd = data_error(fixed.d_obs, ld.system.channel_noise, settings.rel_noise)
        predicted = fixed.d_obs + res * sd
        ok = np.isfinite(res).all(axis=1)
        gains = gains * np.median(fixed.d_obs[ok] / predicted[ok], axis=0)
        gains /= np.exp(np.log(gains).mean())
        history.append(gains.copy())
    return gains, np.array(history)


def _select_job(args):
    ld, stats, settings = args
    out = invert_line(ld, build_prior(stats, settings), settings, seed=1)
    return out["loglik"], np.nanmedian(np.mean(out["coupled"]["residual"] ** 2, axis=1))


def select_settings(ld: LineData, stats: LogStats, base: Settings, grid: dict, workers=4):
    """log p(data | settings) on one line for every combination in grid: highest first.

    The marginal likelihood is what EM maximised in note 05: how probable the
    data are under the prior and noise model, with the earth integrated out.
    The Kalman filter gets it for free, from how well each forecast predicted
    the next sounding.
    """
    keys = list(grid)
    combos = [dict(zip(keys, vals, strict=True)) for vals in product(*grid.values())]
    jobs = [(ld, stats, replace(base, **c)) for c in combos]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_select_job, jobs))
    table = pd.DataFrame(combos)
    table["loglik"] = [r[0] for r in results]
    table["misfit"] = [r[1] for r in results]
    return table.sort_values("loglik", ascending=False, ignore_index=True)


# --- checks -------------------------------------------------------------------------------


def forward_check(con, survey, system, lay, every: int = 13, rel_noise: float = 0.05):
    """Run the physics on EM1DFM's own models and compare with the measured data."""
    res = con.execute(
        f"SELECT sounding_id, layer, resistivity_ohm_m FROM model_resistivity "
        f"WHERE run_id = ? AND sounding_id % {every} = 0",
        [INVERTED_RUN],
    ).df()
    res = res.pivot(index="sounding_id", columns="layer", values="resistivity_ohm_m").sort_index()
    obs = survey.set_index("sounding_id").loc[res.index]
    thickness = lay["thickness_m"].dropna().to_numpy()
    pred = fdem.channels(
        fdem.response(system, 1 / res.to_numpy(), thickness, obs["bird_height_m"].to_numpy())
    )
    d = obs[CHANNEL_COLS].to_numpy()
    sd = data_error(d, system.channel_noise, rel_noise)
    return pd.DataFrame(
        {
            "predicted / measured (median)": np.median(pred / d, axis=0),
            "|residual| / error (median)": np.median(np.abs(pred - d) / sd, axis=0),
            "correlation": [np.corrcoef(pred[:, j], d[:, j])[0, 1] for j in range(d.shape[1])],
        },
        index=CHANNEL_COLS,
    )


SELECTION_LINE = 30050  # no boreholes within 100 m, so tuning on it can't leak their truth
GRID = {  # searched together ...
    "length_m": [1500.0, 3000.0, 6000.0],
    "prior_sd_scale": [1.0, 1.5, 2.0],
    "whole_profile_share": [0.0, 0.4, 0.7],
}
# The likelihood also scores these, at the best of the above, but is not allowed to choose:
# it asks for less noise all the way down to the floors, because the errors that matter
# (1D earth, calibration, footprint) repeat from one sounding to the next, and a likelihood
# that treats soundings as independent can't see that. 5% is how far EM1DFM's own models
# miss the data on most channels (forward_check).
NOISE_GRID = {"rel_noise": [0.01, 0.02, 0.03, 0.05]}


def _finite(depth):
    """inf (no conductor above 125 m) counted as 125 m for errors and maps."""
    return np.where(np.isinf(depth), 125.0, depth)


def results_tables(lds, results, mode="coupled"):
    soundings, layer_rows = [], []
    for ld in lds:
        r = results[ld.line][mode]
        soundings.append(
            pd.DataFrame(
                {
                    "sounding_id": ld.sounding_id, "line": ld.line,
                    "x": ld.xy[:, 0], "y": ld.xy[:, 1], "height_m": ld.height_m,
                    **{k: r[k] for k in (
                        "depth_q10", "depth_q50", "depth_q90", "p_no_conductor", "dh_q50"
                    )},
                    "misfit": np.mean(r["residual"] ** 2, axis=1),
                }
            )
        )  # fmt: skip
        n, k = r["log_rho_q50"].shape
        layer_rows.append(
            pd.DataFrame(
                {
                    "sounding_id": np.repeat(ld.sounding_id, k),
                    "layer": np.tile(np.arange(1, k + 1), n),
                    **{q: r[f"log_rho_{q}"].ravel() for q in ("q10", "q50", "q90")},
                }
            )
        )
    return pd.concat(soundings, ignore_index=True), pd.concat(layer_rows, ignore_index=True)


def nearest(points: pd.DataFrame, xy: np.ndarray, max_dist_m: float = 100.0):
    """For each point (x, y), the index into xy of the nearest step within max_dist_m."""
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(xy).query(points[["x", "y"]].to_numpy())
    keep = dist <= max_dist_m
    return keep, dist, idx


def borehole_table(con, steps, lds, results, prior, settings):
    """Depth to conductor at each borehole that reached bedrock, three ways."""
    holes = con.sql(
        "SELECT a.hole_id, a.bedrock_depth_m, a.bedrock_unit, b.x, b.y "
        "FROM aquifer_base AS a JOIN borehole AS b USING (hole_id) WHERE a.reached_bedrock"
    ).df()
    steps = steps[steps["height_m"] <= MAX_HEIGHT_M].reset_index(drop=True)  # with data only
    keep, dist, idx = nearest(holes, steps[["x", "y"]].to_numpy())
    holes = holes[keep].assign(distance_m=dist[keep], step=idx[keep]).reset_index(drop=True)
    at = steps.iloc[holes["step"]].reset_index(drop=True)
    holes["line"] = at["line"]
    holes["sounding_id"] = at["sounding_id"]
    for q in ("q10", "q50", "q90"):
        holes[f"coupled_{q}"] = at[f"depth_{q}"]

    # the same soundings inverted on their own, without the neighbours
    by_line = {ld.line: ld for ld in lds}
    subsets = []
    for line, grp in holes.groupby("line"):
        ld = by_line[line]
        pos = np.searchsorted(ld.sounding_id, grp["sounding_id"].to_numpy())
        subsets.append((grp.index, ld.subset(pos)))
    indep = invert_lines([s for _, s in subsets], prior, settings, modes=("independent",), seed=7)
    for rows, sub in subsets:
        r = indep[sub.line]["independent"]
        for q in ("q10", "q50", "q90"):
            holes.loc[rows, f"independent_{q}"] = r[f"depth_{q}"]

    em = con.execute(
        "SELECT sounding_id, conductor_top_m FROM model_conductor_depth WHERE run_id = ?",
        [INVERTED_RUN],
    ).df()
    holes["em1dfm"] = holes["sounding_id"].map(em.set_index("sounding_id")["conductor_top_m"])
    holes["inverted_line"] = holes["sounding_id"].isin(em["sounding_id"])
    return holes.drop(columns=["step"]).sort_values(["line", "bedrock_depth_m"])


def inside(truth, lo, hi, tops) -> pd.Series:
    """Does [lo, hi] reach the model layer that holds the true depth?

    Conductor depths can only be layer tops, so a true depth of 6.2 m (in the
    layer from 5.6 to 7.2 m) counts as inside an interval that touches 5.6-7.2.
    And none can be shallower than the first top at or below 1.8 m.
    """
    first = tops[tops >= CONDUCTOR_MIN_TOP_M][0]
    k = np.searchsorted(tops, np.maximum(truth, first), side="right") - 1
    top, bottom = tops[k], np.r_[tops[1:], np.inf][k]
    return pd.Series((lo <= bottom) & (hi >= top), index=truth.index)


def depth_scores(holes: pd.DataFrame, tops: np.ndarray) -> pd.DataFrame:
    rows = {}
    for name, sub in (
        ("inverted lines", holes[holes["inverted_line"]]),
        ("uninverted lines", holes[~holes["inverted_line"]]),
        ("buried valley (bedrock ≥ 40 m)", holes[holes["bedrock_depth_m"] >= 40]),
    ):
        truth = sub["bedrock_depth_m"]
        row = {"holes": len(sub)}
        for mode in ("coupled", "independent"):
            row[f"{mode}: MAE (m)"] = (_finite(sub[f"{mode}_q50"]) - truth).abs().mean()
            hit = inside(truth, sub[f"{mode}_q10"], sub[f"{mode}_q90"], tops)
            row[f"{mode}: inside 80%"] = hit.mean()
        row["EM1DFM: MAE (m)"] = (sub["em1dfm"] - truth).abs().mean()
        rows[name] = row
    return pd.DataFrame(rows).T


def log_comparison(con, steps, layer_table):
    """EnKI at the induction-log holes: does the 80% interval hold the logged resistivity?"""
    logs = con.sql(
        "SELECT l.hole_id, l.layer, log10(l.resistivity_ohm_m) AS log_rho, b.x, b.y "
        "FROM borehole_layer_resistivity AS l JOIN borehole AS b USING (hole_id) "
        "WHERE l.coverage > 0.8"
    ).df()
    holes = logs.drop_duplicates("hole_id")
    steps = steps[steps["height_m"] <= MAX_HEIGHT_M].reset_index(drop=True)  # with data only
    keep, dist, idx = nearest(holes, steps[["x", "y"]].to_numpy())
    near = dict(
        zip(holes["hole_id"][keep], steps["sounding_id"].to_numpy()[idx[keep]], strict=True)
    )
    logs = logs[logs["hole_id"].isin(near)].assign(sounding_id=lambda d: d["hole_id"].map(near))
    both = logs.merge(layer_table, on=["sounding_id", "layer"])
    em = con.execute(
        "SELECT sounding_id, layer, log10(resistivity_ohm_m) AS em1dfm FROM model_resistivity "
        "WHERE run_id = ?",
        [INVERTED_RUN],
    ).df()
    both = both.merge(em, on=["sounding_id", "layer"], how="left")
    from scipy.stats import spearmanr

    return both, {
        "holes": both["hole_id"].nunique(),
        "layer values": len(both),
        "rank correlation, EnKI median vs log": spearmanr(both["q50"], both["log_rho"]).statistic,
        "median (EnKI - log), decades": (both["q50"] - both["log_rho"]).median(),
        "log inside EnKI 80% interval": both["log_rho"].between(both["q10"], both["q90"]).mean(),
        "rank correlation, EM1DFM vs log (where inverted)": spearmanr(
            both["em1dfm"], both["log_rho"], nan_policy="omit"
        ).statistic,
    }


def em1dfm_comparison(con, layer_table, steps):
    em = con.execute(
        "SELECT sounding_id, layer, log10(resistivity_ohm_m) AS em1dfm FROM model_resistivity "
        "WHERE run_id = ?",
        [INVERTED_RUN],
    ).df()
    both = layer_table.merge(em, on=["sounding_id", "layer"])
    tops = con.sql("SELECT layer, top_m FROM model_layer").df()
    both = both.merge(tops, on="layer")
    both["depth band"] = pd.cut(both["top_m"], [-1, 10, 30, 60, 200], labels=[
        "0-10 m", "10-30 m", "30-60 m", "60 m and below"
    ])  # fmt: skip
    table = both.groupby("depth band", observed=True).apply(
        lambda d: pd.Series(
            {
                "median |EnKI - EM1DFM| (decades)": (d["q50"] - d["em1dfm"]).abs().median(),
                "EM1DFM inside EnKI 80%": d["em1dfm"].between(d["q10"], d["q90"]).mean(),
                "EnKI 80% width (decades)": (d["q90"] - d["q10"]).median(),
            }
        ),
        include_groups=False,
    )
    cond = con.execute(
        "SELECT sounding_id, conductor_top_m FROM model_conductor_depth WHERE run_id = ?",
        [INVERTED_RUN],
    ).df()
    s = steps.merge(cond, on="sounding_id").dropna(subset=["conductor_top_m"])
    diff = _finite(s["depth_q50"]) - s["conductor_top_m"]
    return table, {
        "soundings": len(s),
        "median |difference| in depth to conductor (m)": diff.abs().median(),
        "median difference, EnKI - EM1DFM (m)": diff.median(),
        "EM1DFM depth inside EnKI 80%": s["conductor_top_m"]
        .between(s["depth_q10"], s["depth_q90"])
        .mean(),
    }


def run(
    db_path: Path,
    out_dir: Path,
    fig_dir: Path,
    base: Settings | None = None,
    grid: dict | None = None,
    select: bool = True,
) -> dict:
    import time

    from morrill.analysis.kalman import load_survey

    con = duckdb.connect(str(db_path), read_only=True)
    survey = load_survey(con)
    system = fdem.System.from_db(con)
    lay = layers(con)

    check = forward_check(con, survey, system, lay)
    print("1. The physics: EM1DFM's own models run through fdem.py, against the measured data")
    print(check.round(3).to_string())

    near_any = con.sql(
        "SELECT DISTINCT hole_id FROM borehole_sounding WHERE distance_m <= 100"
    ).df()["hole_id"]
    stats = log_stats(con, exclude=tuple(near_any))
    c_free, length_free = stats.fit()
    print(
        "\n2. Induction logs (holes within 100 m of a sounding left out, so the checks stay fair):"
    )
    print(
        f"   {stats.holes} holes; log10 ρ mean {stats.mean:.2f} (≈{10**stats.mean:.0f} ohm·m), "
        f"sd {stats.sd:.2f}; vertical correlation fitted freely: c {c_free:.2f}, "
        f"L_v {length_free:.1f} m"
    )

    base = base or Settings()
    sel = line_data(survey, SELECTION_LINE, base.step, system, lay)
    quick = replace(base, n_ensemble=100)
    t0 = time.perf_counter()
    gains, gain_history = calibrate(sel, stats, quick)
    print(f"\n3. Channel gains on line {SELECTION_LINE} ({time.perf_counter() - t0:.0f} s):")
    print(pd.DataFrame(gain_history, columns=CHANNEL_COLS).round(3).to_string())
    sel = replace(sel, d_obs=sel.d_obs / gains)
    if select:
        t0 = time.perf_counter()
        table = select_settings(sel, stats, quick, grid or GRID)
        best = table.iloc[0][list(grid or GRID)].to_dict()
        noise = select_settings(sel, stats, replace(quick, **best), NOISE_GRID)
        settings = replace(base, **best)
        print(
            f"\n4. Settings by marginal likelihood on line {SELECTION_LINE} "
            f"({len(table) + len(noise)} runs, {time.perf_counter() - t0:.0f} s); top 8:"
        )
        print(table.head(8).round(3).to_string(index=False))
        print(f"   data error, for information (fixed at {base.rel_noise:.0%}; see NOISE_GRID):")
        print(noise.round(3).to_string(index=False))
    else:
        settings = base
        print("\n4. Settings given, not searched")
    prior = build_prior(stats, settings)
    print(f"   chosen: {settings}")
    print("   prior: " + ", ".join(f"{k} {v:.3g}" for k, v in prior.info.items()))

    lines = sorted(survey["line"].unique())
    lds = [line_data(survey, L, settings.step, system, lay) for L in lines]
    lds = [replace(ld, d_obs=ld.d_obs / gains) for ld in lds]
    t0 = time.perf_counter()
    results = invert_lines(lds, prior, settings)
    wall = time.perf_counter() - t0
    n_steps = sum(len(ld.sounding_id) for ld in lds)
    n_obs = sum(int((ld.height_m <= MAX_HEIGHT_M).sum()) for ld in lds)
    print(
        f"\n5. All {len(lines)} lines, every {settings.step}th sounding: {n_steps:,} soundings "
        f"({n_obs:,} updated) in {wall / 60:.1f} min on 4 cores; "
        f"{4 * wall / n_steps:.2f} core-seconds and "
        f"{settings.n_ensemble * settings.n_assim:,} forward models per sounding"
    )

    steps, layer_table = results_tables(lds, results)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(out_dir / "enki_soundings.parquet")
    layer_table.to_parquet(out_dir / "enki_layers.parquet")

    residual = np.concatenate([results[ld.line]["coupled"]["residual"] for ld in lds])
    fit = pd.DataFrame(
        {
            "median residual / error": np.nanmedian(residual, axis=0),
            "median |residual| / error": np.nanmedian(np.abs(residual), axis=0),
        },
        index=CHANNEL_COLS,
    )
    print("\n6. Data fit of the smoothed ensemble mean (residual = predicted - measured):")
    print(fit.round(2).to_string())
    print(
        f"   mean squared residual per sounding: median {steps['misfit'].median():.2f}, "
        f"90th percentile {steps['misfit'].quantile(0.9):.2f} (1 = fits to the assumed error)"
    )

    em_table, em_summary = em1dfm_comparison(con, layer_table, steps)
    print("\n7. Against EM1DFM on the 6 inverted lines:")
    print(em_table.round(2).to_string())
    print("   " + "; ".join(f"{k} {v:.3g}" for k, v in em_summary.items()))

    holes = borehole_table(con, steps, lds, results, prior, settings)
    cols = [
        "hole_id", "line", "distance_m", "bedrock_unit", "bedrock_depth_m", "em1dfm",
        "coupled_q10", "coupled_q50", "coupled_q90",
        "independent_q10", "independent_q50", "independent_q90",
    ]  # fmt: skip
    print(f"\n8. Depth to conductor at the {len(holes)} boreholes within 100 m (inf = none):")
    print(holes[cols].round(1).to_string(index=False))
    holes.to_parquet(out_dir / "enki_boreholes.parquet")
    scores = depth_scores(holes, lay["top_m"].to_numpy())
    print(scores.round(2).to_string())

    both, log_summary = log_comparison(con, steps, layer_table)
    print("\n9. Resistivity at the induction-log holes within 100 m:")
    print("   " + "; ".join(f"{k} {v:.3g}" for k, v in log_summary.items()))

    valley_line, compare_line = draw_figures(con, steps, layer_table, holes, lay, fig_dir)
    print(f"\nfigures: line {valley_line} section, line {compare_line} against EM1DFM, map")
    return {"settings": settings, "gains": gains, "holes": holes, "scores": scores, "steps": steps}


def draw_figures(con, steps, layer_table, holes, lay, fig_dir: Path) -> tuple[int, int]:
    """The three figures for note 06; can be redrawn from the saved parquet files."""
    from morrill.analysis import figures

    fig_dir.mkdir(parents=True, exist_ok=True)
    bedrock = con.sql(
        "SELECT a.hole_id, b.x, b.y, a.bedrock_depth_m FROM aquifer_base AS a "
        "JOIN borehole AS b USING (hole_id) WHERE a.reached_bedrock"
    ).df()
    deep = holes.assign(deep=holes["bedrock_depth_m"] >= 40)
    by_line = (
        deep[~deep["inverted_line"]]
        .groupby("line")
        .agg(deep=("deep", "sum"), n=("hole_id", "size"))
    )
    valley_line = int(by_line.sort_values(["deep", "n"], ascending=False).index[0])
    compare_line = int(holes[holes["inverted_line"]]["line"].value_counts().idxmax())
    em = con.execute(
        "SELECT sounding_id, layer, log10(resistivity_ohm_m) AS em1dfm FROM model_resistivity "
        "WHERE run_id = ?",
        [INVERTED_RUN],
    ).df()
    figures.enki_section(steps, layer_table, lay, holes, valley_line, fig_dir / "enki_line.png")
    figures.enki_compare(
        steps, layer_table, em, lay, holes, compare_line, fig_dir / "enki_vs_em1dfm.png"
    )
    figures.enki_depth_map(steps, bedrock, fig_dir / "enki_conductor_depth.png")
    return valley_line, compare_line
