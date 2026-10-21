"""Depth to bedrock, inverted directly: a sharp contact under a layered fill.

Note 06's smooth model ramps down to a conductor over 10-20 m, so a threshold
on it (the first layer under 20 ohm·m) fires near the top of the ramp. Here
the contact is itself a number in the state:

    θ = [log10 ρ of K fill layers, log10 z_b, log10 ρ of the bedrock, δh]

The fill between the surface and z_b is split into K layers in fixed
proportions (FILL_FRACTIONS of z_b), so every member has the same structure
but its own depth. Below z_b is a bedrock half-space. The ensemble then
carries a distribution of depths directly, and the AR(1) prior along the line
says how fast the contact can rise or fall.

The prior comes from boreholes held out of validation (no hole within 100 m of
a sounding):
- log10 z_b: mean and spread of the depth to bedrock in holes that reached it;
- log10 ρ of fill and bedrock: induction logs above and below the logged
  contact, averaged per hole.

Everything else is enki.py's: the along-line filter, ES-MDA, the smoother,
the channel gains and the data error.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from morrill.analysis import enki, fdem

FILL_FRACTIONS = np.array([0.05, 0.10, 0.15, 0.30, 0.40])  # of z_b, top to bottom
K = len(FILL_FRACTIONS)
LOG_DEPTH_RANGE = (0.0, np.log10(150.0))  # 1 to 150 m
DEPTH, ROCK, HEIGHT = K, K + 1, K + 2  # positions in θ


@dataclass(frozen=True)
class Settings:
    n_ensemble: int = 400  # 100 can lock onto a wrong depth (see the tests)
    n_assim: int = 4
    step: int = 20
    # chosen by marginal likelihood (select_settings); these defaults are its answer
    length_m: float = 3000.0  # L_h along the line
    sd_scale: float = 1.5  # widens the logs' spread of log10 ρ
    depth_sd_scale: float = 2.0  # widens the boreholes' spread of log10 z_b
    rel_noise: float = 0.05  # fixed, as in note 06
    fill_length: float = 1.0  # correlation between fill layers, in layers


def borehole_stats(con: duckdb.DuckDBPyConnection, exclude: tuple[str, ...] = ()) -> dict:
    """Depth to bedrock, and log10 ρ above and below it, from holes not in `exclude`."""
    base = con.sql("SELECT hole_id, bedrock_depth_m FROM aquifer_base WHERE reached_bedrock").df()
    base = base[~base["hole_id"].isin(exclude)]
    lz = np.log10(base["bedrock_depth_m"].clip(lower=1.0))
    logs = con.sql(
        "SELECT l.hole_id, l.depth_m, l.conductivity_ms_m, a.bedrock_depth_m "
        "FROM induction_log_clean AS l JOIN aquifer_base AS a USING (hole_id) "
        "WHERE l.reliable AND a.reached_bedrock"
    ).df()
    logs = logs[~logs["hole_id"].isin(exclude)]
    logs["below"] = logs["depth_m"] >= logs["bedrock_depth_m"]
    # average conductivity per hole and side (current flows through the layers in parallel)
    per_hole = logs.groupby(["hole_id", "below"])["conductivity_ms_m"].mean()
    lr = np.log10(1000.0 / per_hole).unstack()
    return {
        "depth_holes": len(base),
        "log_depth_mean": float(lz.mean()),
        "log_depth_sd": float(lz.std()),
        "log_holes": int(lr.notna().any(axis=1).sum()),
        "fill_mean": float(lr[False].mean()),
        "fill_sd": float(lr[False].std()),
        "rock_mean": float(lr[True].mean()),
        "rock_sd": float(lr[True].std()),
    }


def make_prior(stats: dict, settings: Settings) -> enki.Prior:
    p = K + 3
    cov = np.zeros((p, p))
    i = np.arange(K)
    fill_sd = stats["fill_sd"] * settings.sd_scale
    cov[:K, :K] = fill_sd**2 * np.exp(-np.abs(np.subtract.outer(i, i)) / settings.fill_length)
    cov[DEPTH, DEPTH] = (stats["log_depth_sd"] * settings.depth_sd_scale) ** 2
    cov[ROCK, ROCK] = (stats["rock_sd"] * settings.sd_scale) ** 2
    cov[HEIGHT, HEIGHT] = enki.HEIGHT_SD**2
    mean = np.r_[np.full(K, stats["fill_mean"]), stats["log_depth_mean"], stats["rock_mean"], 0]
    return enki.Prior(mean, cov, dict(stats))


def clip(theta: np.ndarray) -> np.ndarray:
    theta[:, :K] = np.clip(theta[:, :K], *enki.LOG_RHO_RANGE)
    theta[:, DEPTH] = np.clip(theta[:, DEPTH], *LOG_DEPTH_RANGE)
    theta[:, ROCK] = np.clip(theta[:, ROCK], *enki.LOG_RHO_RANGE)
    theta[:, HEIGHT] = np.clip(theta[:, HEIGHT], *enki.HEIGHT_RANGE)
    return theta


def earth(theta: np.ndarray):
    """Conductivities (M, K+1) and per-member thicknesses (M, K) of each member's earth."""
    thickness = 10.0 ** theta[:, DEPTH, None] * FILL_FRACTIONS
    sigma = 10.0 ** -np.c_[theta[:, :K], theta[:, ROCK]]
    return sigma, thickness


def _forwards(ld: enki.LineData, sd: np.ndarray):
    def make(t):
        def g(theta):
            sigma, thickness = earth(theta)
            h = np.maximum(ld.height_m[t] + theta[:, HEIGHT], 5.0)
            return fdem.channels(fdem.response(ld.system, sigma, thickness, h)) / sd[t]

        return g

    return [make(t) for t in range(len(ld.d_obs))]


def summarise(ens: np.ndarray) -> dict[str, np.ndarray]:
    depth = 10.0 ** ens[..., DEPTH]
    dq = np.quantile(depth, [0.1, 0.5, 0.9], axis=1)
    rq = np.quantile(ens[..., ROCK], [0.1, 0.5, 0.9], axis=1)
    fill = np.median(ens[..., :K], axis=1)  # (n, K)
    return {
        "depth_q10": dq[0], "depth_q50": dq[1], "depth_q90": dq[2],
        "log_rho_rock_q10": rq[0], "log_rho_rock_q50": rq[1], "log_rho_rock_q90": rq[2],
        "log_rho_fill_q50": fill.mean(axis=1), "log_rho_fill_top_q50": fill[:, 0],
        "dh_q50": np.median(ens[..., HEIGHT], axis=1),
    }  # fmt: skip


def invert_line(ld: enki.LineData, prior: enki.Prior, settings: Settings, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sd = enki.data_error(ld.d_obs, ld.system.channel_noise, settings.rel_noise)
    Y = ld.d_obs / sd
    observed = ld.height_m <= enki.MAX_HEIGHT_M
    forwards = _forwards(ld, sd)
    F, A, logpdf = enki.filter_line(
        Y, forwards, ld.step_m, observed, prior, settings, rng, clip_fn=clip
    )
    S = enki.rts_smoother(F, A)
    S = clip(S.reshape(-1, S.shape[-1])).reshape(S.shape)  # the smoother can step past bounds
    return {
        **summarise(S),
        "residual": enki._residuals(S, Y, forwards, observed),
        "loglik": np.nansum(logpdf) - np.log(sd[observed]).sum(),
    }


def _job(args):
    ld, prior, settings, seed = args
    return invert_line(ld, prior, settings, seed)


def invert_lines(lines_data, prior, settings, workers=4, seed=0):
    jobs = [(ld, prior, settings, seed + ld.line) for ld in lines_data]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(zip([ld.line for ld in lines_data], pool.map(_job, jobs), strict=True))


def _select_job(args):
    ld, stats, settings = args
    out = invert_line(ld, make_prior(stats, settings), settings, seed=1)
    return out["loglik"], np.nanmedian(np.mean(out["residual"] ** 2, axis=1))


GRID = {
    "length_m": [1000.0, 3000.0],
    "sd_scale": [1.0, 1.5],
    "depth_sd_scale": [1.0, 1.5, 2.0],
}


def select_settings(ld, stats, base: Settings, grid: dict, workers=4) -> pd.DataFrame:
    """log p(data | settings) on one line, highest first (as enki.select_settings)."""
    combos = [dict(zip(grid, v, strict=True)) for v in product(*grid.values())]
    jobs = [(ld, stats, replace(base, **c)) for c in combos]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_select_job, jobs))
    table = pd.DataFrame(combos)
    table["loglik"] = [r[0] for r in results]
    table["misfit"] = [r[1] for r in results]
    return table.sort_values("loglik", ascending=False, ignore_index=True)


# --- checks and outputs -------------------------------------------------------------------


def results_table(lds, results) -> pd.DataFrame:
    keys = [k for k in next(iter(results.values())) if k not in ("residual", "loglik")]
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "sounding_id": ld.sounding_id, "line": ld.line, "x": ld.xy[:, 0],
                    "y": ld.xy[:, 1], "height_m": ld.height_m,
                    **{k: results[ld.line][k] for k in keys},
                    "misfit": np.mean(results[ld.line]["residual"] ** 2, axis=1),
                }
            )
            for ld in lds
        ],
        ignore_index=True,
    )  # fmt: skip


def borehole_table(
    con, steps: pd.DataFrame, smooth: pd.DataFrame | None = None, near=0.0, far=100.0
) -> pd.DataFrame:
    """Each borehole that reached bedrock whose nearest step with data is near-far m away.

    0-100 m gives the validation holes; 100-300 m the calibration holes (see `inflation`).
    """
    holes = con.sql(
        "SELECT a.hole_id, a.bedrock_depth_m, a.bedrock_unit, b.x, b.y "
        "FROM aquifer_base AS a JOIN borehole AS b USING (hole_id) WHERE a.reached_bedrock"
    ).df()
    with_data = steps[steps["height_m"] <= enki.MAX_HEIGHT_M].reset_index(drop=True)
    keep, dist, idx = enki.nearest(holes, with_data[["x", "y"]].to_numpy(), far)
    keep &= dist > near
    holes = holes[keep].assign(distance_m=dist[keep]).reset_index(drop=True)
    at = with_data.iloc[idx[keep]].reset_index(drop=True)
    for c in ("line", "sounding_id", "depth_q10", "depth_q50", "depth_q90", "misfit"):
        holes[c] = at[c].to_numpy()
    em = con.execute(
        "SELECT sounding_id, conductor_top_m FROM model_conductor_depth WHERE run_id = ?",
        [enki.INVERTED_RUN],
    ).df()
    holes["inverted_line"] = holes["sounding_id"].isin(em["sounding_id"])
    holes["em1dfm"] = holes["sounding_id"].map(em.set_index("sounding_id")["conductor_top_m"])
    if smooth is not None:
        sm = smooth.set_index("hole_id")
        for c in ("coupled_q10", "coupled_q50", "coupled_q90"):
            holes["smooth_" + c.split("_")[1]] = holes["hole_id"].map(sm[c])
    return holes.sort_values(["line", "bedrock_depth_m"], ignore_index=True)


def conformity(holes: pd.DataFrame) -> np.ndarray:
    """How far each borehole depth lies from the median, in log depth, in units of that
    side of the 80% interval: 1 is on its edge, so s <= 1 means inside."""
    truth = np.log10(holes["bedrock_depth_m"].clip(*10.0 ** np.array(LOG_DEPTH_RANGE)))
    lo, mid, hi = (np.log10(holes[c]) for c in ("depth_q10", "depth_q50", "depth_q90"))
    return np.where(truth < mid, (mid - truth) / (mid - lo), (truth - mid) / (hi - mid))


def inflation(scores: np.ndarray, level: float = 0.8) -> float:
    """Split-conformal factor k: widening both sides of the interval k-fold covers a new
    hole with probability >= level, if it is exchangeable with the calibration holes."""
    rank = int(np.ceil((len(scores) + 1) * level))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else np.inf


def widen(q10, q50, q90, k: float):
    """The 80% interval with each side stretched k-fold in log depth."""
    lo, mid, hi = np.log10(q10), np.log10(q50), np.log10(q90)
    return 10.0 ** (mid - k * (mid - lo)), 10.0 ** (mid + k * (hi - mid))


def depth_scores(holes: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "inverted lines": holes["inverted_line"],
        "uninverted lines": ~holes["inverted_line"],
        "buried valley (bedrock ≥ 40 m)": holes["bedrock_depth_m"] >= 40,
        "all": holes["bedrock_depth_m"].notna(),
    }
    rows = {}
    for name, sel in groups.items():
        sub = holes[sel]
        truth = sub["bedrock_depth_m"]
        row = {
            "holes": len(sub),
            "sharp: mean error (m)": (sub["depth_q50"] - truth).abs().mean(),
            "sharp: median error (m)": (sub["depth_q50"] - truth).abs().median(),
            "sharp: inside 80%": truth.between(sub["depth_q10"], sub["depth_q90"]).mean(),
        }
        if "depth_lo" in sub:
            row["sharp: inside recalibrated 80%"] = truth.between(
                sub["depth_lo"], sub["depth_hi"]
            ).mean()
        if "smooth_q50" in sub:
            row["smooth: mean error (m)"] = (enki._finite(sub["smooth_q50"]) - truth).abs().mean()
        row["EM1DFM: mean error (m)"] = (sub["em1dfm"] - truth).abs().mean()
        rows[name] = row
    return pd.DataFrame(rows).T


def run(
    db_path: Path,
    out_dir: Path,
    fig_dir: Path,
    base: Settings | None = None,
    grid: dict | None = None,
    select: bool = True,
) -> dict:
    import time

    from morrill.analysis import figures
    from morrill.analysis.kalman import load_survey

    con = duckdb.connect(str(db_path), read_only=True)
    survey = load_survey(con)
    system = fdem.System.from_db(con)
    lay = enki.layers(con)
    near_any = tuple(
        con.sql("SELECT DISTINCT hole_id FROM borehole_sounding WHERE distance_m <= 100").df()[
            "hole_id"
        ]
    )
    stats = borehole_stats(con, exclude=near_any)
    print("1. Prior from boreholes with no sounding within 100 m:")
    print(
        f"   depth to bedrock: {stats['depth_holes']} holes, median "
        f"{10 ** stats['log_depth_mean']:.1f} m, sd {stats['log_depth_sd']:.2f} decades\n"
        f"   logs ({stats['log_holes']} holes): fill {10 ** stats['fill_mean']:.0f} ohm·m "
        f"(sd {stats['fill_sd']:.2f}), bedrock {10 ** stats['rock_mean']:.0f} ohm·m "
        f"(sd {stats['rock_sd']:.2f})"
    )

    base = base or Settings()
    sel = enki.line_data(survey, enki.SELECTION_LINE, base.step, system, lay)
    t0 = time.perf_counter()
    gains, _ = enki.calibrate(
        sel, enki.log_stats(con, exclude=near_any), enki.Settings(n_ensemble=100)
    )
    print(f"\n2. Channel gains, as in note 06 ({time.perf_counter() - t0:.0f} s):")
    print("   " + " ".join(f"{g:.3f}" for g in gains))
    sel = replace(sel, d_obs=sel.d_obs / gains)
    if select:
        t0 = time.perf_counter()
        # 200 members: with 100, depth and fill resistivity can trade off into a wrong answer
        table = select_settings(sel, stats, replace(base, n_ensemble=200), grid or GRID)
        base = replace(base, **table.iloc[0][list(grid or GRID)].to_dict())
        print(
            f"\n3. Settings by marginal likelihood on line {enki.SELECTION_LINE} "
            f"({len(table)} runs, {time.perf_counter() - t0:.0f} s):"
        )
        print(table.round(3).to_string(index=False))
    settings = base
    prior = make_prior(stats, settings)
    print(f"   chosen: {settings}")

    lines = sorted(survey["line"].unique())
    lds = [enki.line_data(survey, L, settings.step, system, lay) for L in lines]
    lds = [replace(ld, d_obs=ld.d_obs / gains) for ld in lds]
    t0 = time.perf_counter()
    results = invert_lines(lds, prior, settings)
    wall = time.perf_counter() - t0
    n = sum(len(ld.sounding_id) for ld in lds)
    print(
        f"\n4. All {len(lines)} lines: {n:,} soundings in {wall / 60:.1f} min on 4 cores "
        f"({4 * wall / n:.2f} core-seconds per sounding)"
    )
    steps = results_table(lds, results)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(out_dir / "bedrock_soundings.parquet")
    residual = np.concatenate([results[ld.line]["residual"] for ld in lds])
    print(
        f"   data misfit per sounding: median {steps['misfit'].median():.2f}, "
        f"90th percentile {steps['misfit'].quantile(0.9):.2f}"
    )
    print(
        "   median residual / error by channel: "
        + " ".join(f"{v:+.2f}" for v in np.nanmedian(residual, axis=0))
    )

    smooth_path = out_dir / "enki_boreholes.parquet"
    smooth = pd.read_parquet(smooth_path) if smooth_path.exists() else None
    # the posterior covers uncertainty given the two-unit structure, not the error of the
    # structure itself; calibrate on holes 100-300 m out, which validation never sees
    calib = borehole_table(con, steps, near=100.0, far=300.0)
    calib["score"] = conformity(calib)
    k = inflation(calib["score"].to_numpy())
    print("\n5a. Calibration holes (nearest step 100-300 m away; not used for validation):")
    print(
        calib[["hole_id", "line", "distance_m", "bedrock_depth_m", "depth_q10", "depth_q50",
               "depth_q90", "score"]].round(2).to_string(index=False)
    )  # fmt: skip
    print(
        f"   {(calib['score'] <= 1).mean():.0%} inside the 80% interval; split-conformal "
        f"factor k = {k:.2f} from {len(calib)} holes"
    )
    calib.to_parquet(out_dir / "bedrock_calibration.parquet")

    holes = borehole_table(con, steps, smooth)
    holes["depth_lo"], holes["depth_hi"] = widen(
        holes["depth_q10"], holes["depth_q50"], holes["depth_q90"], k
    )
    holes.to_parquet(out_dir / "bedrock_boreholes.parquet")
    cols = [
        "hole_id", "line", "distance_m", "bedrock_unit", "bedrock_depth_m",
        "depth_q10", "depth_q50", "depth_q90", "depth_lo", "depth_hi", "smooth_q50", "em1dfm",
        "misfit",
    ]  # fmt: skip
    print(f"\n5b. Depth to bedrock at the {len(holes)} boreholes within 100 m:")
    print(holes[[c for c in cols if c in holes]].round(1).to_string(index=False))
    scores = depth_scores(holes)
    print(scores.round(2).to_string())

    fig_dir.mkdir(parents=True, exist_ok=True)
    bedrock = con.sql(
        "SELECT a.hole_id, b.x, b.y, a.bedrock_depth_m FROM aquifer_base AS a "
        "JOIN borehole AS b USING (hole_id) WHERE a.reached_bedrock"
    ).df()
    line = 30070
    smooth_steps_path = out_dir / "enki_soundings.parquet"
    smooth_layers_path = out_dir / "enki_layers.parquet"
    if smooth_steps_path.exists() and smooth_layers_path.exists():
        figures.bedrock_section(
            steps, pd.read_parquet(smooth_steps_path), pd.read_parquet(smooth_layers_path),
            lay, holes, line, fig_dir / "bedrock_line.png", k=k,
        )  # fmt: skip
    figures.bedrock_map(steps, bedrock, fig_dir / "bedrock_depth.png")
    print(f"\nfigures: line {line} section, map")
    return {"settings": settings, "holes": holes, "scores": scores, "steps": steps, "k": k}
