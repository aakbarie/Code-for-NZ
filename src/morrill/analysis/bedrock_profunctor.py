"""Depth to bedrock as a profunctor: exact along the line, on a grid of depths.

Note 07's ensemble keeps one Gaussian bump per sounding. Where the data allow
a shallow and a deep contact, it keeps whichever its seed lands on. Here the
depth lives on a grid of n_depth values, and composition along the line is
exact (profunctor.py), so two answers stay two answers.

The earth and its prior are note 07's: five fill layers in fixed proportions
of z_b, a bedrock half-space, a bird-height correction, and the prior from
held-out boreholes. Only z_b is carried along the line. The other seven
numbers ψ are integrated out at every grid depth of every sounding:

    L_t(z) = ∫ p(d_t | z, ψ) p(ψ) dψ      (composition through ψ)

The integral uses a Laplace approximation around a Levenberg-Marquardt fit,
warm-started from the same depth at the previous sounding. That is the one
approximation. The other difference from note 07 is that ψ is not coupled
along the line.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from morrill.analysis import bedrock, enki, fdem, profunctor

NU = [*range(bedrock.K), bedrock.ROCK, bedrock.HEIGHT]  # the 7 numbers integrated out
EPS = np.r_[np.full(bedrock.K + 1, 1e-3), 1e-2]  # finite-difference steps: log10 ρ, m


@dataclass(frozen=True)
class Settings:
    n_depth: int = 80  # log-spaced, 1-150 m: 0.027 decades apart, half a posterior sd
    # note 07's choices, so the two inversions share one Bayesian model
    length_m: float = 3000.0
    sd_scale: float = 1.5
    depth_sd_scale: float = 1.0
    rel_noise: float = 0.05
    fill_length: float = 1.0
    step: int = 20
    iters_first: int = 12  # Levenberg-Marquardt steps at the first sounding
    iters: int = 4  # later soundings start from the previous one's fit
    restart_every: int = 10  # also fit from the prior mean, keeping the better
    n_samples: int = 400


def depth_grid(settings: Settings) -> np.ndarray:
    """log10 z_b at the grid nodes."""
    return np.linspace(*bedrock.LOG_DEPTH_RANGE, settings.n_depth)


def _theta(psi, log_z):
    theta = np.empty((len(psi), bedrock.K + 3))
    theta[:, NU] = psi
    theta[:, bedrock.DEPTH] = log_z
    return theta


def laplace(resid, psi, log_z, m, P_inv, logdet_P, lo, hi, n_iter: int):
    """Fit ψ at each depth by Levenberg-Marquardt, then approximate

        log ∫ N(ψ; m, P) exp(-|r(ψ)|² / 2) dψ  ≈  -f(ψ*) - (log|P| + log|H|) / 2,

    with f the penalised misfit and H = JᵀJ + P⁻¹ at the fit ψ*.
    resid(psi, log_z) gives whitened residuals (B, n). Returns ψ*, log evidence, r(ψ*).
    """
    B, p = psi.shape
    idx = np.arange(p)

    def objective(r, psi):
        d = psi - m
        return 0.5 * (np.sum(r**2, axis=1) + np.einsum("bi,ij,bj->b", d, P_inv, d))

    def jacobian(psi, r):
        pert = (psi[:, None, :] + np.diag(EPS)[None]).reshape(-1, p)
        rp = resid(pert, np.repeat(log_z, p)).reshape(B, p, -1)
        return np.transpose((rp - r[:, None, :]) / EPS[None, :, None], (0, 2, 1))

    r = resid(psi, log_z)
    f = objective(r, psi)
    lam = np.full(B, 1e-2)
    for _ in range(n_iter):
        J = jacobian(psi, r)
        g = np.einsum("bnp,bn->bp", J, r) + (psi - m) @ P_inv
        H = np.einsum("bnp,bnq->bpq", J, J) + P_inv
        D = H.copy()
        D[:, idx, idx] *= 1 + lam[:, None]
        trial = np.clip(psi - np.linalg.solve(D, g[..., None])[..., 0], lo, hi)
        r_trial = resid(trial, log_z)
        f_trial = objective(r_trial, trial)
        ok = f_trial < f
        psi = np.where(ok[:, None], trial, psi)
        r = np.where(ok[:, None], r_trial, r)
        f = np.minimum(f, f_trial)
        lam = np.clip(np.where(ok, lam / 3, lam * 5), 1e-6, 1e6)
    J = jacobian(psi, r)
    _, logdet_H = np.linalg.slogdet(np.einsum("bnp,bnq->bpq", J, J) + P_inv)
    return psi, -f - 0.5 * (logdet_P + logdet_H), r


def evidence(ld: enki.LineData, prior: enki.Prior, settings: Settings):
    """log L_t(z) on the depth grid (n, k), the misfit there, and the fitted ψ."""
    x = depth_grid(settings)
    sd = enki.data_error(ld.d_obs, ld.system.channel_noise, settings.rel_noise)
    y = ld.d_obs / sd
    observed = ld.height_m <= enki.MAX_HEIGHT_M
    forwards = bedrock._forwards(ld, sd)
    m = prior.mean[NU]
    P = prior.cov[np.ix_(NU, NU)]
    P_inv, logdet_P = np.linalg.inv(P), np.linalg.slogdet(P)[1]
    lo = np.r_[np.full(bedrock.K + 1, enki.LOG_RHO_RANGE[0]), enki.HEIGHT_RANGE[0]]
    hi = np.r_[np.full(bedrock.K + 1, enki.LOG_RHO_RANGE[1]), enki.HEIGHT_RANGE[1]]
    n, k = len(y), len(x)
    log_L, misfit = np.zeros((n, k)), np.full((n, k), np.nan)
    psi_fit = np.full((n, k, len(NU)), np.nan)
    psi, fitted = np.tile(m, (k, 1)), 0
    for t in np.flatnonzero(observed):

        def resid(p_, lz, t=t):
            return forwards[t](_theta(p_, lz)) - y[t]

        iters = settings.iters_first if fitted == 0 else settings.iters
        psi, ev, r = laplace(resid, psi, x, m, P_inv, logdet_P, lo, hi, iters)
        if fitted and fitted % settings.restart_every == 0:  # guard against a stale start
            psi2, ev2, r2 = laplace(
                resid, np.tile(m, (k, 1)), x, m, P_inv, logdet_P, lo, hi, settings.iters_first
            )
            better = ev2 > ev
            psi = np.where(better[:, None], psi2, psi)
            r = np.where(better[:, None], r2, r)
            ev = np.where(better, ev2, ev)
        fitted += 1
        # the data's own normalisation, so evidences compare across models
        log_L[t] = ev - np.log(sd[t]).sum() - 0.5 * len(y[t]) * np.log(2 * np.pi)
        misfit[t] = np.mean(r**2, axis=1)
        psi_fit[t] = psi
    return log_L, misfit, psi_fit


def depth_prior(prior: enki.Prior) -> tuple[float, float]:
    return prior.mean[bedrock.DEPTH], float(np.sqrt(prior.cov[bedrock.DEPTH, bedrock.DEPTH]))


def kernels(step_m: np.ndarray, x, log_prior, mean: float, sd: float, length_m: float):
    """One log-kernel (k, k) per step from t to t+1, from the AR(1) correlation exp(-Δ/L_h)."""
    return np.array([
        profunctor.reversible_kernel(x, log_prior, mean, sd, np.exp(-d / length_m))
        for d in step_m[1:]
    ])  # fmt: skip


def compose_line(log_L, step_m, x, mean, sd, length_m):
    """The along-line composite: log posterior (n, k), best path, evidence, kernels."""
    log_prior = profunctor.grid_log_prior(x, mean, sd)
    K_ = kernels(step_m, x, log_prior, mean, sd, length_m)
    log_post, ev = profunctor.posterior(log_prior, K_, log_L)
    return log_post, profunctor.best_path(log_prior, K_, log_L), ev, K_, log_prior


def summarise_line(ld, fit, x, mean, sd, length_m, n_samples, seed) -> dict:
    """Compose one line from its per-sounding evidence: posterior, best surface, samples."""
    log_L, misfit, psi_fit = fit
    log_post, path, ev, K_, log_prior = compose_line(log_L, ld.step_m, x, mean, sd, length_m)
    fwd, _, _ = profunctor.chain(log_prior, K_, log_L)
    rng = np.random.default_rng(seed)
    idx = profunctor.sample_paths(fwd, K_, rng, n_samples)
    dx = x[1] - x[0]
    members = 10.0 ** (x[idx] + rng.uniform(-dx / 2, dx / 2, idx.shape))
    q = 10.0 ** profunctor.quantiles(log_post, x)
    rows = np.arange(len(path))
    return {
        "depth_q10": q[0], "depth_q50": q[1], "depth_q90": q[2],
        "depth_best": 10.0 ** x[path],
        "n_answers": profunctor.n_modes(log_post),
        "misfit": misfit[rows, path],
        "log_rho_fill_best": psi_fit[rows, path, : bedrock.K].mean(axis=1),
        "log_rho_rock_best": psi_fit[rows, path, bedrock.K],
        "members": np.clip(members, *10.0 ** np.array(bedrock.LOG_DEPTH_RANGE)).astype(np.float32),
        "log_post": log_post.astype(np.float32),
        "loglik": ev,
    }  # fmt: skip


def invert_line(ld: enki.LineData, prior: enki.Prior, settings: Settings, seed: int) -> dict:
    mean, sd = depth_prior(prior)
    fit = evidence(ld, prior, settings)
    return summarise_line(
        ld, fit, depth_grid(settings), mean, sd, settings.length_m, settings.n_samples, seed
    )


def _evidence_job(args):
    return evidence(*args)


def evidence_lines(lds, prior, settings, workers=4) -> dict:
    """The expensive part, per line in parallel: log L_t(z), misfit and fitted ψ."""
    jobs = sorted(lds, key=lambda ld: -len(ld.sounding_id))  # longest first
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = pool.map(_evidence_job, [(ld, prior, settings) for ld in jobs])
        return dict(zip([ld.line for ld in jobs], out, strict=True))


KEYS = ("depth_q10", "depth_q50", "depth_q90", "depth_best", "n_answers", "misfit",
        "log_rho_fill_best", "log_rho_rock_best")  # fmt: skip


def results_table(lds, results) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "sounding_id": ld.sounding_id,
                    "line": ld.line,
                    "x": ld.xy[:, 0],
                    "y": ld.xy[:, 1],
                    "height_m": ld.height_m,
                    **{c: results[ld.line][c] for c in KEYS},
                }
            )  # fmt: skip
            for ld in lds
        ],
        ignore_index=True,
    )


GRID = {"length_m": [1000.0, 3000.0, 6000.0, 12000.0, 24000.0], "depth_sd_scale": [0.75, 1.0, 1.5]}


def select(lds, fits, prior, settings, grid=GRID) -> pd.DataFrame:
    """Log evidence summed over all lines for each L_h and depth-prior width. L_t(z) depends
    on neither, so the search needs only the composition, not new fits."""
    x = depth_grid(settings)
    mean, sd = depth_prior(prior)
    rows = []
    for length_m in grid["length_m"]:
        for scale in grid["depth_sd_scale"]:
            s = sd / settings.depth_sd_scale * scale
            ev = sum(
                compose_line(fits[ld.line][0], ld.step_m, x, mean, s, length_m)[2] for ld in lds
            )
            rows.append({"length_m": length_m, "depth_sd_scale": scale, "log_evidence": ev})
    return pd.DataFrame(rows).sort_values("log_evidence", ascending=False, ignore_index=True)


def run(db_path: Path, out_dir: Path, fig_dir: Path, settings: Settings | None = None) -> dict:
    from morrill.analysis import aquifer, figures
    from morrill.analysis.kalman import load_survey

    settings = settings or Settings()
    con = duckdb.connect(str(db_path), read_only=True)
    survey = load_survey(con)
    system = fdem.System.from_db(con)
    lay = enki.layers(con)
    near_any = tuple(
        con.sql("SELECT DISTINCT hole_id FROM borehole_sounding WHERE distance_m <= 100").df()[
            "hole_id"
        ]
    )
    stats = bedrock.borehole_stats(con, exclude=near_any)
    bs = bedrock.Settings(
        length_m=settings.length_m, sd_scale=settings.sd_scale,
        depth_sd_scale=settings.depth_sd_scale, rel_noise=settings.rel_noise,
        fill_length=settings.fill_length,
    )  # fmt: skip
    prior = bedrock.make_prior(stats, bs)
    sel = enki.line_data(survey, enki.SELECTION_LINE, settings.step, system, lay)
    gains, _ = enki.calibrate(
        sel, enki.log_stats(con, exclude=near_any), enki.Settings(n_ensemble=100)
    )
    lines = sorted(survey["line"].unique())
    lds = [enki.line_data(survey, L, settings.step, system, lay) for L in lines]
    lds = [replace(ld, d_obs=ld.d_obs / gains) for ld in lds]
    print(f"1. Note 07's prior, settings and channel gains; {settings.n_depth} depths, 1-150 m")

    t0 = time.perf_counter()
    fits = evidence_lines(lds, prior, settings)
    wall = time.perf_counter() - t0
    n = sum(len(ld.sounding_id) for ld in lds)
    print(
        f"\n2. Evidence on the grid, all {len(lines)} lines: {n:,} soundings in "
        f"{wall / 60:.1f} min on 4 cores ({4 * wall / n:.2f} core-seconds per sounding)"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "profunctor_evidence.npz",
        sounding_id=np.concatenate([ld.sounding_id for ld in lds]),
        log_L=np.concatenate([fits[ld.line][0] for ld in lds]),
        log_depth=depth_grid(settings),
        # the fitted ψ and misfit at every grid depth, for links beyond the depth (coarse.py)
        psi=np.concatenate([fits[ld.line][2] for ld in lds]).astype(np.float32),
        misfit=np.concatenate([fits[ld.line][1] for ld in lds]).astype(np.float32),
    )

    table = select(lds, fits, prior, settings)
    best = table.iloc[0]
    print("\n3. Log evidence over all lines, by along-line length and depth-prior width:")
    print(table.round(1).to_string(index=False))
    x = depth_grid(settings)
    mean, sd = depth_prior(prior)
    sd = sd / settings.depth_sd_scale * best["depth_sd_scale"]
    settings = replace(
        settings, length_m=float(best["length_m"]), depth_sd_scale=float(best["depth_sd_scale"])
    )
    print(f"   chosen: L_h = {settings.length_m:.0f} m, depth spread × {settings.depth_sd_scale}")
    results = {
        ld.line: summarise_line(
            ld, fits[ld.line], x, mean, sd, settings.length_m, settings.n_samples, ld.line
        )
        for ld in lds
    }
    steps = results_table(lds, results)
    steps.to_parquet(out_dir / "profunctor_soundings.parquet")
    np.savez_compressed(
        out_dir / "profunctor_members.npz",
        sounding_id=steps["sounding_id"].to_numpy(),
        depth_m=np.concatenate([results[ld.line]["members"] for ld in lds]),
        log_post=np.concatenate([results[ld.line]["log_post"] for ld in lds]),
        log_depth=x,
    )
    print(
        f"   data misfit at the best surface: median {steps['misfit'].median():.2f}, "
        f"90th percentile {steps['misfit'].quantile(0.9):.2f}"
    )
    multi = steps["n_answers"] >= 2
    print(f"   soundings whose posterior holds two or more answers: {multi.mean():.1%}")
    print("   by line: " + ", ".join(
        f"{L} {v:.0%}" for L, v in steps.assign(m=multi).groupby("line")["m"].mean().items()
    ))  # fmt: skip

    calib = bedrock.borehole_table(con, steps, near=100.0, far=300.0)
    calib["score"] = bedrock.conformity(calib)
    k = bedrock.inflation(calib["score"].to_numpy())
    calib.to_parquet(out_dir / "profunctor_calibration.parquet")
    print(
        f"\n4. Calibration holes (100-300 m): {(calib['score'] <= 1).mean():.0%} inside the "
        f"80% interval; split-conformal factor k = {k:.2f} from {len(calib)} holes"
    )
    smooth_path = out_dir / "enki_boreholes.parquet"
    holes = bedrock.borehole_table(
        con, steps, pd.read_parquet(smooth_path) if smooth_path.exists() else None
    )
    holes["depth_lo"], holes["depth_hi"] = bedrock.widen(
        holes["depth_q10"], holes["depth_q50"], holes["depth_q90"], k
    )
    at = steps.set_index("sounding_id")
    holes["depth_best"] = holes["sounding_id"].map(at["depth_best"])
    holes["n_answers"] = holes["sounding_id"].map(at["n_answers"])
    ens_path = out_dir / "bedrock_boreholes.parquet"
    if ens_path.exists():
        ens = pd.read_parquet(ens_path).set_index("hole_id")
        for c in ("depth_q10", "depth_q50", "depth_q90"):
            holes["ensemble_" + c.split("_")[1]] = holes["hole_id"].map(ens[c])
    holes.to_parquet(out_dir / "profunctor_boreholes.parquet")
    cols = [
        "hole_id", "line", "bedrock_depth_m", "depth_q10", "depth_q50", "depth_q90",
        "depth_best", "n_answers", "ensemble_q50", "smooth_q50", "misfit",
    ]  # fmt: skip
    print(f"\n5. Depth to bedrock at the {len(holes)} validation holes:")
    print(holes[[c for c in cols if c in holes]].round(1).to_string(index=False))
    scores = bedrock.depth_scores(holes)
    print(scores.round(2).to_string())

    fig_dir.mkdir(parents=True, exist_ok=True)
    post = np.load(out_dir / "profunctor_members.npz")
    ens_steps = out_dir / "bedrock_soundings.parquet"
    figures.profunctor_lines(
        steps, post["log_post"], post["log_depth"],
        pd.read_parquet(ens_steps) if ens_steps.exists() else None,
        holes, (30090, 30070), fig_dir / "profunctor_lines.png",
    )  # fmt: skip
    figures.profunctor_map(steps, fig_dir / "profunctor_map.png")
    print("\nfigures: lines 30090 and 30070, map\n\n6. Aquifer material from the sampled surfaces:")
    vol = aquifer.run(db_path, out_dir, fig_dir, source="profunctor")
    return {"steps": steps, "holes": holes, "scores": scores, "k": k, "select": table, **vol}
