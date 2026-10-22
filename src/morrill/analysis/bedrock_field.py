"""Depth to bedrock across the block: lines composed with each other, and boreholes as data.

Note 08 composed each flight line exactly, on its own. Neighbouring lines are
about 200 m apart, close compared with the distance over which the bedrock
surface changes, so each line knows something about the next. Composing the
whole 2D field exactly is out of reach: the cost grows as 80 to the power of
the number of lines. Each line's whole surface given its neighbours is still
a chain, which note 08 composes exactly. Drawing line after line that way is a
Gibbs sampler whose blocks are whole lines, and it converges to the exact
posterior of the 2D model.

The 2D model adds one kind of link, between a sounding and its nearest
sounding on each neighbouring line (and where tie lines cross). Across such a
link log10 z_b changes by a Gaussian amount whose variance grows with distance
at the rate the along-line evidence chose: v = 2σ²/L_h per metre (isotropy).

Boreholes enter as one more link: a logged depth at its nearest sounding. Its
error is not fitted against the EM. Bedrock depth is rough at the scale of a
single hole, and pairs of holes measure that directly: the variance of the
difference in log10 depth between two holes d metres apart is c0 + b·d. That
is the error of a point measurement of a surface the bird averages over about
100 m, and it is estimated from holes that are not validation holes.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from morrill.analysis import bedrock, bedrock_profunctor, enki, fdem, profunctor


@dataclass(frozen=True)
class Settings:
    length_m: float = 24000.0  # note 08's choice by evidence
    depth_sd_scale: float = 1.5
    across_scale: float = 1.0  # × the isotropic across-line variance
    max_tie_m: float = 100.0  # tie-line crossings
    sweeps: int = 300
    burn: int = 100
    chains: int = 2
    icm_sweeps: int = 20


@dataclass
class Line:
    rows: np.ndarray  # positions in the survey-wide arrays
    log_L: np.ndarray  # (n, k) per-sounding evidence
    log_kernels: np.ndarray  # (n-1, k, k)
    log_prior: np.ndarray  # (k,)


@dataclass
class Edges:
    """Undirected links, stored in both directions: (i, j, variance of the change)."""

    i: np.ndarray
    j: np.ndarray
    var: np.ndarray


def _pairs(a_xy, b_xy, max_d):
    d, idx = cKDTree(b_xy).query(a_xy)
    keep = d <= max_d
    return np.flatnonzero(keep), idx[keep], d[keep]


def across_edges(steps: pd.DataFrame, ns_lines, tie_lines, v: float, settings: Settings):
    """Each sounding to its nearest on each neighbouring line (within 1.5 line spacings),
    and tie-line soundings to the nearest on each north-south line (within max_tie_m)."""
    xy = steps[["x", "y"]].to_numpy()
    line = steps["line"].to_numpy()
    rows = {L: np.flatnonzero(line == L) for L in [*ns_lines, *tie_lines]}
    east = {L: np.median(xy[rows[L], 0]) for L in ns_lines}
    order = sorted(ns_lines, key=east.get)
    found = set()

    def link(A, B, max_d):
        for a, b in ((A, B), (B, A)):
            ia, ib, _ = _pairs(xy[rows[a]], xy[rows[b]], max_d)
            for p, q in zip(rows[a][ia], rows[b][ib], strict=True):
                found.add((min(p, q), max(p, q)))

    for A, B in zip(order[:-1], order[1:], strict=True):
        link(A, B, 1.5 * abs(east[B] - east[A]))
    for T in tie_lines:
        for L in ns_lines:
            link(T, L, settings.max_tie_m)
    i, j = np.array(sorted(found)).T
    d = np.hypot(*(xy[i] - xy[j]).T)
    var = settings.across_scale * v * np.maximum(d, 1.0)
    return Edges(np.r_[i, j], np.r_[j, i], np.r_[var, var])


class Field:
    """Lines and the links between them, with a per-sounding extra term (boreholes)."""

    def __init__(self, lines: list[Line], edges: Edges, x: np.ndarray, n_total: int):
        self.lines, self.x, self.n = lines, x, n_total
        self.fast = [profunctor.Line(ln.log_prior, ln.log_kernels) for ln in lines]
        self.local = []
        for ln in lines:
            pos = np.full(n_total, -1)
            pos[ln.rows] = np.arange(len(ln.rows))
            m = pos[edges.i] >= 0
            self.local.append((pos[edges.i[m]], edges.j[m], edges.var[m]))

    def unary(self, li: int, z: np.ndarray, extra: np.ndarray) -> np.ndarray:
        ln = self.lines[li]
        U = ln.log_L + extra[ln.rows]
        i_loc, j, var = self.local[li]
        if len(j):
            d = self.x[None, :] - self.x[z[j]][:, None]
            np.add.at(U, i_loc, -0.5 * d**2 / var[:, None])
        return U

    def gibbs(self, extra, z0, sweeps: int, burn: int, rng, keep: bool = True):
        """Line-block Gibbs. Returns Rao-Blackwellised log marginals (n, k): the average of
        each line's exact conditional marginals over the kept sweeps; the kept joint states
        (n, S); and the final state."""
        z = z0.copy()
        k = len(self.x)
        acc = np.zeros((self.n, k))
        kept = []
        for sweep in range(sweeps):
            for li in rng.permutation(len(self.lines)):
                ln, fast = self.lines[li], self.fast[li]
                fwd, bwd, _ = fast.compose(self.unary(li, z, extra), backward=sweep >= burn)
                if sweep >= burn:
                    acc[ln.rows] += fast.marginals(fwd, bwd)
                z[ln.rows] = fast.sample(fwd, rng)
            if sweep >= burn and keep:
                kept.append(z.copy())
        with np.errstate(divide="ignore"):
            log_marg = np.log(acc / max(sweeps - burn, 1))
        return log_marg, (np.array(kept).T if kept else None), z

    def icm(self, extra, z0, sweeps: int) -> np.ndarray:
        """Block coordinate ascent in max-plus: each line's best path given its neighbours."""
        z = z0.copy()
        for _ in range(sweeps):
            before = z.copy()
            for li, ln in enumerate(self.lines):
                z[ln.rows] = profunctor.best_path(
                    ln.log_prior, ln.log_kernels, self.unary(li, z, extra)
                )
            if np.array_equal(z, before):
                break
        return z


def hole_variogram(con, exclude=(), max_d: float = 400.0) -> tuple[float, float]:
    """Variance of the difference in log10 depth to bedrock between two holes d metres
    apart, as c0 + b·d, fitted to pairs closer than max_d from holes not in exclude."""
    h = con.sql(
        "SELECT a.hole_id, a.bedrock_depth_m AS z, b.x, b.y FROM aquifer_base AS a "
        "JOIN borehole AS b USING (hole_id) WHERE a.reached_bedrock"
    ).df()
    h = h[~h["hole_id"].isin(exclude)]
    lz = np.log10(h["z"].clip(lower=1.0)).to_numpy()
    xy = h[["x", "y"]].to_numpy()
    i, j = np.triu_indices(len(h), 1)
    d = np.hypot(*(xy[i] - xy[j]).T)
    sel = d < max_d
    c0, b = np.linalg.lstsq(
        np.c_[np.ones(sel.sum()), d[sel]], (lz[i] - lz[j])[sel] ** 2, rcond=None
    )[0]
    return float(c0), float(b)


def point_predictive(log_marg_row, x, var: float) -> np.ndarray:
    """Log density over the grid of a hole's log depth: the footprint-scale posterior
    blurred by the point-scale variability (variance var)."""
    return logsumexp(log_marg_row[None, :] - 0.5 * (x[:, None] - x[None, :]) ** 2 / var, axis=1)


def borehole_terms(holes: pd.DataFrame, row_of: dict, x, c0: float, b: float, n: int):
    """Each hole as a Gaussian link to its nearest sounding: log N(log z_hole; x, c0 + b·d)."""
    extra = np.zeros((n, len(x)))
    for _, h in holes.iterrows():
        var = c0 + b * h["distance_m"]
        lz = np.log10(max(h["bedrock_depth_m"], 1.0))
        extra[row_of[h["sounding_id"]]] += -0.5 * (x - lz) ** 2 / var - 0.5 * np.log(
            2 * np.pi * var
        )
    return extra


def fit_hole_error(holes: pd.DataFrame, row_of: dict, log_marg, x, v: float, grid=None):
    """s0 maximising the predictive density of the holes' log depths under log_marg."""
    grid = np.linspace(0.02, 0.6, 59) if grid is None else grid
    lz = np.log10(holes["bedrock_depth_m"].clip(lower=1.0)).to_numpy()
    lp = np.array([log_marg[row_of[s]] for s in holes["sounding_id"]])
    d = holes["distance_m"].to_numpy()

    def score(s0):
        var = (s0**2 + v * d)[:, None]
        return float(np.sum(logsumexp(lp - 0.5 * (x[None] - lz[:, None]) ** 2 / var, axis=1)
                            - 0.5 * np.log(2 * np.pi * var[:, 0])))  # fmt: skip

    scores = np.array([score(s) for s in grid])
    return float(grid[scores.argmax()]), scores


def summarise(log_marg, x) -> dict:
    q = 10.0 ** profunctor.quantiles(log_marg, x)
    return {"depth_q10": q[0], "depth_q50": q[1], "depth_q90": q[2]}


def run(db_path: Path, out_dir: Path, fig_dir: Path, settings: Settings | None = None) -> dict:
    import time

    from morrill.analysis import aquifer, figures
    from morrill.analysis.kalman import load_survey

    settings = settings or Settings()
    rng = np.random.default_rng(0)
    con = duckdb.connect(str(db_path), read_only=True)
    ev = np.load(out_dir / "profunctor_evidence.npz")
    steps = pd.read_parquet(out_dir / "profunctor_soundings.parquet")
    assert np.array_equal(ev["sounding_id"], steps["sounding_id"].to_numpy())
    one_d = np.load(out_dir / "profunctor_members.npz")
    x, log_L = ev["log_depth"], ev["log_L"]
    n = len(steps)
    row_of = dict(zip(steps["sounding_id"], range(n), strict=True))

    near_any = tuple(
        con.sql("SELECT DISTINCT hole_id FROM borehole_sounding WHERE distance_m <= 100").df()[
            "hole_id"
        ]
    )
    stats = bedrock.borehole_stats(con, exclude=near_any)
    mean = stats["log_depth_mean"]
    sd = stats["log_depth_sd"] * settings.depth_sd_scale
    v = 2 * sd**2 / settings.length_m
    survey = load_survey(con)
    system = fdem.System.from_db(con)
    lay = enki.layers(con)
    lines, all_lines = [], sorted(steps["line"].unique())
    log_prior = profunctor.grid_log_prior(x, mean, sd)
    for L in all_lines:
        ld = enki.line_data(survey, L, 20, system, lay)
        rows = np.array([row_of[s] for s in ld.sounding_id])
        K_ = bedrock_profunctor.kernels(ld.step_m, x, log_prior, mean, sd, settings.length_m)
        lines.append(Line(rows, log_L[rows], K_, log_prior))
    ns = [L for L in all_lines if L < 39000]
    ties = [L for L in all_lines if L >= 39000]
    edges = across_edges(steps, ns, ties, v, settings)
    field = Field(lines, edges, x, n)
    print(
        f"1. The field: {len(lines)} lines, {n:,} soundings, {len(edges.i) // 2:,} links between "
        f"lines; across-line spread {np.sqrt(v * 200):.3f} decades at 200 m (isotropic)"
    )

    def index(depth_m):
        return np.clip(np.rint((np.log10(depth_m) - x[0]) / (x[1] - x[0])), 0, len(x) - 1).astype(
            int
        )

    no_holes = np.zeros((n, len(x)))
    t0 = time.perf_counter()
    margs, kept, finals = [], [], []
    for c in range(settings.chains):
        z0 = index(one_d["depth_m"][:, c * 37])  # different 1D surfaces as starts
        lm, kp, zf = field.gibbs(no_holes, z0, settings.sweeps, settings.burn, rng)
        margs.append(lm)
        kept.append(kp)
        finals.append(zf)
    wall = time.perf_counter() - t0
    log_marg = logsumexp(np.stack(margs), axis=0) - np.log(len(margs))
    q_chain = [10 ** profunctor.quantiles(m, x)[1] for m in margs]
    gap = np.abs(np.log10(q_chain[0]) - np.log10(q_chain[-1]))
    print(
        f"\n2. Line-block Gibbs, {settings.chains} chains × {settings.sweeps} sweeps "
        f"({settings.burn} burn-in): {wall / 60:.1f} min. Chains agree on the median log "
        f"depth to {np.median(gap):.3f} decades (median), {np.quantile(gap, 0.99):.3f} (99th %)"
    )
    best = field.icm(no_holes, index(steps["depth_best"].to_numpy()), settings.icm_sweeps)
    field_steps = steps[["sounding_id", "line", "x", "y", "height_m"]].assign(
        **summarise(log_marg, x), depth_best=10.0 ** x[best],
        n_answers=profunctor.n_modes(log_marg), misfit=steps["misfit"],
    )  # fmt: skip
    field_steps.to_parquet(out_dir / "field_soundings.parquet")
    dx = x[1] - x[0]
    states = np.concatenate(kept, axis=1)
    members = 10.0 ** (x[states] + rng.uniform(-dx / 2, dx / 2, states.shape))
    np.savez_compressed(
        out_dir / "field_members.npz", sounding_id=field_steps["sounding_id"].to_numpy(),
        depth_m=members.astype(np.float32), log_marg=log_marg.astype(np.float32), log_depth=x,
    )  # fmt: skip

    calib = bedrock.borehole_table(con, field_steps, near=100.0, far=300.0)
    calib["score"] = bedrock.conformity(calib)
    k = bedrock.inflation(calib["score"].to_numpy())
    calib.to_parquet(out_dir / "field_calibration.parquet")
    holes = bedrock.borehole_table(con, field_steps)
    holes["depth_lo"], holes["depth_hi"] = bedrock.widen(
        holes["depth_q10"], holes["depth_q50"], holes["depth_q90"], k
    )
    one = pd.read_parquet(out_dir / "profunctor_boreholes.parquet").set_index("hole_id")
    holes["line_only_q50"] = holes["hole_id"].map(one["depth_q50"])
    print(
        f"   calibration holes inside the 80% interval: {(calib['score'] <= 1).mean():.0%}; "
        f"split-conformal k = {k:.2f}"
    )

    sens = {}
    for scale in (0.5, 2.0):
        e2 = Edges(edges.i, edges.j, edges.var * scale)
        lm_s, _, _ = Field(lines, e2, x, n).gibbs(no_holes, finals[0], 150, 50, rng, False)
        q = 10.0 ** profunctor.quantiles(lm_s[[row_of[s_] for s_ in holes["sounding_id"]]], x)
        t = holes["bedrock_depth_m"].to_numpy()
        sens[scale] = ((np.abs(q[1] - t)).mean(), ((t >= q[0]) & (t <= q[2])).mean())
    print("   sensitivity, across-line variance ×0.5 / ×2: mean error "
          f"{sens[0.5][0]:.2f} / {sens[2.0][0]:.2f} m, inside 80% "
          f"{sens[0.5][1]:.0%} / {sens[2.0][1]:.0%}")  # fmt: skip

    # the point-scale variability of the bedrock surface, from pairs of non-validation holes
    c0, b = hole_variogram(con, exclude=tuple(holes["hole_id"]))
    s0, _ = fit_hole_error(calib, row_of, log_marg, x, v)
    rows_v = np.array([row_of[s_] for s_ in holes["sounding_id"]])

    def point_quantiles(lm_rows, dist):
        dens = np.array([
            point_predictive(r, x, c0 + b * d) for r, d in zip(lm_rows, dist, strict=True)
        ])  # fmt: skip
        return 10.0 ** profunctor.quantiles(dens, x)

    pq = point_quantiles(log_marg[rows_v], holes["distance_m"])
    for c, vals in zip(("point_depth_q10", "point_depth_q50", "point_depth_q90"), pq, strict=True):
        holes[c] = vals
    truth = holes["bedrock_depth_m"]
    em_in = truth.between(holes["depth_q10"], holes["depth_q90"]).mean()
    pt_in = truth.between(holes["point_depth_q10"], holes["point_depth_q90"]).mean()
    print(
        f"\n3. Point against footprint: holes d m apart differ in log10 depth with variance "
        f"{c0:.4f} + {b:.2e}·d (pairs < 400 m, no validation hole).\n"
        f"   Validation holes inside the 80% interval: EM field alone {em_in:.0%}; with the "
        f"point-scale variability added {pt_in:.0%}.\n"
        f"   (Fitting the EM-hole disagreement alone would put s0 = {s0:.2f} decades on each hole.)"
    )

    # boreholes as data, with that error
    cal_terms = borehole_terms(calib, row_of, x, c0, b, n)
    z_start = finals[0]
    lm_cal, _, _ = field.gibbs(cal_terms, z_start, settings.sweeps, settings.burn, rng, False)
    cq = point_quantiles(lm_cal[rows_v], holes["distance_m"])
    for c, vals in zip(("cal_depth_q10", "cal_depth_q50", "cal_depth_q90"), cq, strict=True):
        holes[c] = vals

    # leave one validation hole out, with every other hole (calibration and validation) as data
    all_terms = cal_terms + borehole_terms(holes, row_of, x, c0, b, n)
    loo = np.empty((3, len(holes)))
    t0 = time.perf_counter()
    for h in range(len(holes)):
        mine = borehole_terms(holes.iloc[[h]], row_of, x, c0, b, n)
        lm_h, _, _ = field.gibbs(all_terms - mine, z_start, 150, 50, rng, False)
        loo[:, h] = point_quantiles(lm_h[rows_v[h] : rows_v[h] + 1], [holes["distance_m"].iloc[h]])[
            :, 0
        ]
    print(f"   leave-one-out over the {len(holes)} validation holes: "
          f"{(time.perf_counter() - t0) / 60:.1f} min")  # fmt: skip
    for c, vals in zip(("loo_depth_q10", "loo_depth_q50", "loo_depth_q90"), loo, strict=True):
        holes[c] = vals
    holes.to_parquet(out_dir / "field_boreholes.parquet")

    def score(prefix):
        truth = holes["bedrock_depth_m"]
        q10, q50, q90 = (holes[prefix + c] for c in ("depth_q10", "depth_q50", "depth_q90"))
        valley = truth >= 40
        return {
            "mean error (m)": (q50 - truth).abs().mean(),
            "median error (m)": (q50 - truth).abs().median(),
            "valley mean error (m)": (q50 - truth)[valley].abs().mean(),
            "inside 80%": truth.between(q10, q90).mean(),
        }

    table = pd.DataFrame({
        "lines alone (note 08)": {
            "mean error (m)": (holes["line_only_q50"] - holes["bedrock_depth_m"]).abs().mean()
        },
        "2D field (footprint interval)": score(""),
        "2D field + point-scale variability": score("point_"),
        "2D field + 15 calibration holes as data": score("cal_"),
        "2D field + all other holes as data (leave one out)": score("loo_"),
    }).T  # fmt: skip
    print("\n4. The 21 validation holes:")
    print(table.round(2).to_string())
    cols = ["hole_id", "line", "bedrock_depth_m", "line_only_q50", "depth_q50", "cal_depth_q50",
            "loo_depth_q10", "loo_depth_q50", "loo_depth_q90"]  # fmt: skip
    print(holes[cols].round(1).to_string(index=False))

    fig_dir.mkdir(parents=True, exist_ok=True)
    figures.field_lines(steps, field_steps, holes, (30080, 30090), fig_dir / "field_lines.png")
    print(
        "\nfigures: lines 30080 and 30090\n\n5. Aquifer material from the field's joint surfaces:"
    )
    vol = aquifer.run(db_path, out_dir, fig_dir, source="field")
    return {
        "steps": field_steps,
        "holes": holes,
        "table": table,
        "k": k,
        "variogram": (c0, b),
        **vol,
    }
