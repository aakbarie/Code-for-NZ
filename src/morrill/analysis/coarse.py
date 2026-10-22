"""From the bedrock posterior to coarse sediment: the fill's resistivity, read as lithology.

At every grid depth, the sharp-contact model has fitted the resistivity of five
fill layers (bedrock_profunctor.evidence). Note 04's histogram model turns
resistivity into P(coarse / fine / bedrock). It has to be trained on the same
inversion it is applied to (note 04), so here it is retrained on these fill
layers, against the borehole lithology inside each one.

Composing the two relations, the earth's depth to bedrock and the lithology
given resistivity, gives the thickness of coarse sediment (sand and gravel)
above bedrock at every sounding. Its uncertainty comes from the sampled
bedrock surfaces, each with the fill fitted at that depth.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from morrill.analysis import aquifer, bedrock, lithology

BOUNDS = np.r_[0.0, np.cumsum(bedrock.FILL_FRACTIONS)]  # fill layer tops and bottoms, × z_b


def node_of(depth_m, x) -> np.ndarray:
    return np.clip(np.rint((np.log10(depth_m) - x[0]) / (x[1] - x[0])), 0, len(x) - 1).astype(int)


def training_pairs(con, steps, psi, x, max_dist_m: float = 200.0) -> pd.DataFrame:
    """One row per (hole, fill layer) at the nearest step's median depth to bedrock: the
    fitted log10 ρ of that layer and the metres of each class logged inside it."""
    holes = con.sql("SELECT hole_id, x, y FROM borehole WHERE has_lithology ORDER BY hole_id").df()
    dist, idx = cKDTree(steps[["x", "y"]].to_numpy()).query(holes[["x", "y"]].to_numpy())
    holes = holes.assign(row=idx, distance_m=dist)[dist <= max_dist_m]
    lith = con.sql("SELECT hole_id, from_m, to_m, texture FROM lithology_classified").df()
    lith["cls"] = lith["texture"].map(lithology.CLASS_OF_TEXTURE)
    lith = lith.dropna(subset=["cls"])
    rows = []
    for _, h in holes.iterrows():
        z = steps["depth_q50"].iloc[h["row"]]
        node = node_of(z, x)
        mine = lith[lith["hole_id"] == h["hole_id"]]
        for layer in range(bedrock.K):
            top, bottom = BOUNDS[layer] * z, BOUNDS[layer + 1] * z
            overlap = (np.minimum(mine["to_m"], bottom) - np.maximum(mine["from_m"], top)).clip(0)
            metres = overlap.groupby(mine["cls"]).sum()
            if metres.sum() <= 0:
                continue
            rows.append({
                "hole_id": h["hole_id"], "sounding_id": steps["sounding_id"].iloc[h["row"]],
                "distance_m": h["distance_m"], "layer": layer + 1, "top_m": top,
                "bottom_m": bottom, "log_rho": float(psi[h["row"], node, layer]),
                **{c: float(metres.get(c, 0.0)) for c in lithology.CLASSES},
            })  # fmt: skip
    return pd.DataFrame(rows)


def leave_one_hole_out(pairs: pd.DataFrame, **model_kw) -> np.ndarray:
    out = np.empty((len(pairs), len(lithology.CLASSES)))
    depth = ((pairs["top_m"] + pairs["bottom_m"]) / 2).to_numpy()
    for hole in pairs["hole_id"].unique():
        test = (pairs["hole_id"] == hole).to_numpy()
        model = lithology.HistogramModel(superposition=False, **model_kw).fit(pairs[~test])
        out[test] = model.predict_cells(pairs.loc[test, "log_rho"].to_numpy(), depth[test])
    return out


def coarse_thickness(model, psi, x, nodes) -> np.ndarray:
    """Metres of coarse sediment above bedrock for each sounding and sampled surface:
    Σ over fill layers of P(coarse) × layer thickness, with the fill fitted at that depth.
    nodes: (n, S) grid indices of the sampled depths."""
    n, S = nodes.shape
    z = 10.0 ** x[nodes]  # (n, S)
    out = np.zeros((n, S))
    for layer in range(bedrock.K):
        thick = (BOUNDS[layer + 1] - BOUNDS[layer]) * z
        mid = (BOUNDS[layer] + BOUNDS[layer + 1]) / 2 * z
        log_rho = psi[np.arange(n)[:, None], nodes, layer]
        p = model.predict_cells(log_rho.ravel(), mid.ravel())[:, 0].reshape(n, S)
        out += p * thick
    return out


def run(db_path: Path, out_dir: Path, fig_dir: Path, source: str = "field") -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    ev = np.load(out_dir / "profunctor_evidence.npz")
    x, psi = ev["log_depth"], ev["psi"]
    steps = pd.read_parquet(out_dir / f"{source}_soundings.parquet")
    assert np.array_equal(ev["sounding_id"], steps["sounding_id"].to_numpy())

    pairs = training_pairs(con, steps, psi, x)
    proba = leave_one_hole_out(pairs)
    sc = lithology.scores(pairs, proba)
    print(
        f"1. Lithology from the fill's resistivity, one hole left out at a time "
        f"({pairs['hole_id'].nunique()} holes, {len(pairs)} fill layers):\n"
        f"   accuracy {sc['accuracy']:.2f}; aquifer (coarse) yes/no {sc['aquifer_accuracy']:.2f}, "
        f"against {lithology.cutoff_baseline(pairs):.2f} for the 20 ohm·m rule; "
        f"recall coarse {sc['recall_coarse']:.2f}, fine {sc['recall_fine']:.2f}"
    )
    print(lithology.calibration(pairs, proba).to_string())

    model = lithology.HistogramModel(superposition=False).fit(pairs)
    members = np.load(out_dir / f"{source}_members.npz")["depth_m"]
    nodes = node_of(members, x)
    coarse = coarse_thickness(model, psi, x, nodes)
    total = members.astype(float)
    widths = aquifer.strip_widths(
        steps[steps["line"].isin(aquifer.NS_LINES)].groupby("line")["x"].median()
    )
    rows, vol = [], {"coarse": 0.0, "all": 0.0}
    draws = {"coarse": 0.0, "all": 0.0}
    for line in aquifer.NS_LINES:
        sel = np.flatnonzero(steps["line"].to_numpy() == line)
        order = sel[np.argsort(steps["y"].to_numpy()[sel])]
        w = aquifer.strip_weights(steps["y"].to_numpy()[order])
        area = w.sum() * widths[line]
        c_line = w @ coarse[order] / w.sum()
        a_line = w @ total[order] / w.sum()
        draws["coarse"] = draws["coarse"] + area * c_line / 1e9
        draws["all"] = draws["all"] + area * a_line / 1e9
        rows.append({
            "line": line, "coarse_m_q50": np.median(c_line),
            "coarse_m_q10": np.quantile(c_line, 0.1), "coarse_m_q90": np.quantile(c_line, 0.9),
            "all_m_q50": np.median(a_line), "coarse_share": np.median(c_line / a_line),
        })  # fmt: skip
    table = pd.DataFrame(rows)
    for key in vol:
        vol[key] = np.quantile(draws[key], [0.1, 0.5, 0.9])
    print("\n2. Mean thickness along each line (m): coarse sediment, and all above bedrock")
    print(table.round(2).to_string(index=False))
    print(
        f"\n3. Volume under the ten lines' strips (km³), median [80%]: coarse "
        f"{vol['coarse'][1]:.3f} [{vol['coarse'][0]:.3f}-{vol['coarse'][2]:.3f}], "
        f"all material {vol['all'][1]:.3f} [{vol['all'][0]:.3f}-{vol['all'][2]:.3f}]"
    )
    steps.assign(coarse_m_q50=np.median(coarse, axis=1)).to_parquet(
        out_dir / f"{source}_coarse.parquet"
    )
    return {"pairs": pairs, "scores": sc, "table": table, "volumes": vol}
