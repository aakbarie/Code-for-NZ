"""Rock class from resistivity: a Bayesian histogram model.

For each class c (coarse sediment, fine sediment, bedrock):
    likelihood  P(log ρ | c)  = histogram of inverted resistivity where boreholes say c
    prior       P(c | depth)  = share of logged metres that are c, at that depth
    posterior   P(c | ρ, depth) ∝ likelihood × prior

Superposition: in this valley loose sediment lies on bedrock. Only 3 of the 90
holes that reach bedrock log sand or gravel beneath it (21 m in all). So once
a profile's posterior says "probably bedrock" (P >= 0.5), the layers below use
a "below bedrock" prior learned from the boreholes instead of the depth prior.
It is soft: a strongly sand-like resistivity can still win.

Training pairs come from boreholes within 200 m of an inverted sounding: for
each model layer, how many metres of each class the driller logged inside it
(soft labels: a layer that is half sand, half clay counts half to each).
The histograms are built from the INVERSION's resistivities, not the
borehole logs, because the model is applied to inversion values; the two
differ in scale and bias (note 03).
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

CLASSES = ("coarse", "fine", "bedrock")
BEDROCK = CLASSES.index("bedrock")
CLASS_OF_TEXTURE = {
    "gravel": "coarse",
    "sand_gravel": "coarse",
    "sand": "coarse",
    "silty_sand": "coarse",
    "silt": "fine",
    "clay": "fine",
    "mudrock": "bedrock",
}  # sandstone, ash, soil, fill, unknown: too few metres to model; left out
BINS = np.linspace(0.0, 3.0, 31)  # log10 ohm·m, 1-1000 in 0.1-decade bins
DEPTH_BANDS = (0.0, 5.0, 15.0, 30.0, 1e9)
MAX_LAYER = 21  # top 83.7 m, as in the community analysis


def load_profiles(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    """log10 resistivity (clipped to 1-1000 ohm·m): rows = soundings, columns = layers."""
    df = con.execute(
        """
        SELECT sounding_id, layer, log10(least(greatest(resistivity_ohm_m, 1), 1000)) AS lr
        FROM model_resistivity
        WHERE run_id = ? AND layer <= ? AND sounding_id IS NOT NULL
        """,
        [run_id, MAX_LAYER],
    ).df()
    return df.pivot(index="sounding_id", columns="layer", values="lr").sort_index()


def layer_mid_depths(con: duckdb.DuckDBPyConnection) -> np.ndarray:
    rows = con.execute(
        "SELECT (top_m + bottom_m) / 2 FROM model_layer WHERE layer <= ? ORDER BY layer",
        [MAX_LAYER],
    ).fetchall()
    return np.array([r[0] for r in rows])


def training_pairs(
    con: duckdb.DuckDBPyConnection, run_id: str = "em1dfm_pt1", max_dist_m: float = 200.0
) -> pd.DataFrame:
    """One row per (hole, layer): inverted log10 resistivity and logged metres per class."""
    inverted = con.execute(
        "SELECT DISTINCT m.sounding_id, s.x, s.y FROM model_resistivity m "
        "JOIN sounding s USING (sounding_id) WHERE m.run_id = ? ORDER BY 1",
        [run_id],
    ).df()
    holes = con.sql("SELECT hole_id, x, y FROM borehole WHERE has_lithology ORDER BY hole_id").df()
    dist, idx = cKDTree(inverted[["x", "y"]].to_numpy()).query(holes[["x", "y"]].to_numpy())
    near = holes.assign(sounding_id=inverted["sounding_id"].to_numpy()[idx], distance_m=dist)[
        dist <= max_dist_m
    ]

    con.register("_near", near[["hole_id", "sounding_id", "distance_m"]])
    con.register(
        "_classes", pd.DataFrame(list(CLASS_OF_TEXTURE.items()), columns=["texture", "cls"])
    )
    pairs = con.execute(
        """
        SELECT n.hole_id, n.sounding_id, n.distance_m, l.layer, l.top_m, l.bottom_m,
               log10(least(greatest(m.resistivity_ohm_m, 1), 1000)) AS log_rho,
               k.cls,
               sum(least(c.to_m, l.bottom_m) - greatest(c.from_m, l.top_m)) AS metres
        FROM _near AS n
        JOIN model_layer AS l ON l.layer <= ?
        JOIN model_resistivity AS m
          ON m.run_id = ? AND m.sounding_id = n.sounding_id AND m.layer = l.layer
        JOIN lithology_classified AS c
          ON c.hole_id = n.hole_id AND c.from_m < l.bottom_m AND c.to_m > l.top_m
        JOIN _classes AS k USING (texture)
        GROUP BY ALL
        """,
        [MAX_LAYER, run_id],
    ).df()
    con.unregister("_near")
    con.unregister("_classes")
    wide = pairs.pivot_table(
        index=["hole_id", "sounding_id", "distance_m", "layer", "top_m", "bottom_m", "log_rho"],
        columns="cls",
        values="metres",
        fill_value=0.0,
    )
    wide = wide.reindex(columns=list(CLASSES), fill_value=0.0).reset_index()
    return wide.sort_values(["hole_id", "layer"]).reset_index(drop=True)


def depth_band(depth_m: np.ndarray) -> np.ndarray:
    return np.digitize(depth_m, DEPTH_BANDS[1:-1])


def _bin(log_rho: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(np.clip(log_rho, 0, 3), BINS) - 1, 0, len(BINS) - 2)


class HistogramModel:
    """P(class | log ρ, depth) from weighted histograms and depth-dependent priors.

    smooth_bins     Gaussian smoothing of each histogram, in bins.
    floor           share of each likelihood spread evenly over all bins, so
                    resistivities never seen in training fall back to the prior.
    pseudo_metres   how strongly each depth band's prior is pulled toward the
                    overall prior (shrinkage, for bands with little data).
    by_depth        use the depth-band prior (else one overall prior).
    superposition   below the first layer with P(bedrock) >= trigger, use the
                    below-bedrock prior.
    use_resistivity False gives the prior alone, as a baseline.
    """

    def __init__(
        self,
        smooth_bins: float = 1.5,
        floor: float = 0.01,
        pseudo_metres: float = 20.0,
        by_depth: bool = True,
        superposition: bool = True,
        trigger: float = 0.7,
        use_resistivity: bool = True,
    ):
        self.smooth_bins = smooth_bins
        self.floor = floor
        self.pseudo_metres = pseudo_metres
        self.by_depth = by_depth
        self.superposition = superposition
        self.trigger = trigger
        self.use_resistivity = use_resistivity

    def _shrunk(self, metres: np.ndarray, overall: np.ndarray) -> np.ndarray:
        return (metres + self.pseudo_metres * overall) / (metres.sum() + self.pseudo_metres)

    def fit(self, pairs: pd.DataFrame) -> "HistogramModel":
        metres = pairs[list(CLASSES)].to_numpy()
        log_rho = pairs["log_rho"].to_numpy()
        n_bins = len(BINS) - 1

        self.likelihood_ = np.empty((len(CLASSES), n_bins))
        for c in range(len(CLASSES)):
            h, _ = np.histogram(np.clip(log_rho, 0, 3), bins=BINS, weights=metres[:, c])
            h = gaussian_filter1d(h / h.sum(), self.smooth_bins, mode="nearest")
            self.likelihood_[c] = (1 - self.floor) * h / h.sum() + self.floor / n_bins

        overall = metres.sum(axis=0) / metres.sum()
        self.prior_ = np.tile(overall, (len(DEPTH_BANDS) - 1, 1))
        if self.by_depth:
            band = depth_band(((pairs["top_m"] + pairs["bottom_m"]) / 2).to_numpy())
            for b in range(len(DEPTH_BANDS) - 1):
                self.prior_[b] = self._shrunk(metres[band == b].sum(axis=0), overall)

        # metres logged below each hole's first mostly-bedrock layer
        share = metres / metres.sum(axis=1, keepdims=True)
        first = (
            pairs.assign(is_bedrock=share[:, BEDROCK] >= 0.5)
            .query("is_bedrock")
            .groupby("hole_id")["layer"]
            .min()
        )
        below = pairs["layer"].to_numpy() > pairs["hole_id"].map(first).fillna(np.inf).to_numpy()
        self.prior_below_bedrock_ = self._shrunk(metres[below].sum(axis=0), overall)
        return self

    def predict_cells(self, log_rho: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Each cell on its own (no superposition)."""
        post = self.prior_[depth_band(depth_m)]
        if self.use_resistivity:
            post = post * self.likelihood_[:, _bin(log_rho)].T
        return post / post.sum(axis=1, keepdims=True)

    def predict_profiles(self, log_rho: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """log_rho: (soundings, layers) ordered top down; depth_m: (layers,). -> (s, l, classes)."""
        n, k = log_rho.shape
        out = self.predict_cells(log_rho.ravel(), np.tile(depth_m, n)).reshape(n, k, -1)
        if not (self.superposition and self.use_resistivity):
            return out
        reached = np.zeros(n, dtype=bool)
        for j in range(k):
            if reached.any():
                post = self.prior_below_bedrock_ * self.likelihood_[:, _bin(log_rho[reached, j])].T
                out[reached, j] = post / post.sum(axis=1, keepdims=True)
            reached |= out[:, j, BEDROCK] >= self.trigger
        return out


def leave_one_hole_out(
    pairs: pd.DataFrame, profiles: pd.DataFrame, depth_m: np.ndarray, **model_kw
) -> np.ndarray:
    """Predict each hole's layers from a model fitted on all the other holes."""
    out = np.empty((len(pairs), len(CLASSES)))
    for hole in pairs["hole_id"].unique():
        test = (pairs["hole_id"] == hole).to_numpy()
        model = HistogramModel(**model_kw).fit(pairs[~test])
        sid = pairs.loc[test, "sounding_id"].iloc[0]
        proba = model.predict_profiles(profiles.loc[[sid]].to_numpy(), depth_m)[0]
        out[test] = proba[pairs.loc[test, "layer"].to_numpy() - 1]
    return out


def scores(pairs: pd.DataFrame, proba: np.ndarray) -> dict[str, float]:
    """Accuracy on layers that are at least 60% one class; log loss on soft labels."""
    metres = pairs[list(CLASSES)].to_numpy()
    share = metres / metres.sum(axis=1, keepdims=True)
    clear = share.max(axis=1) >= 0.6
    truth, guess = share.argmax(axis=1)[clear], proba.argmax(axis=1)[clear]
    recall = {c: float(np.mean(guess[truth == i] == i)) for i, c in enumerate(CLASSES)}
    coarse_truth = share[:, 0] >= 0.5
    return {
        "accuracy": float(np.mean(truth == guess)),
        **{f"recall_{c}": r for c, r in recall.items()},
        "log_loss": float(-np.mean(np.sum(share * np.log(np.clip(proba, 1e-9, 1)), axis=1))),
        "aquifer_accuracy": float(np.mean((proba[:, 0] >= 0.5) == coarse_truth)),
    }


def cutoff_baseline(pairs: pd.DataFrame, ohm_m: float = 20.0) -> float:
    """The 2012 binary rule: aquifer where resistivity >= 20 ohm·m."""
    metres = pairs[list(CLASSES)].to_numpy()
    coarse_truth = metres[:, 0] / metres.sum(axis=1) >= 0.5
    return float(np.mean((pairs["log_rho"].to_numpy() >= np.log10(ohm_m)) == coarse_truth))


def calibration(pairs: pd.DataFrame, proba: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Predicted P(coarse) against the observed share of coarse metres, by probability bin."""
    metres = pairs[list(CLASSES)].to_numpy()
    observed = metres[:, 0] / metres.sum(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(proba[:, 0], edges) - 1, 0, n_bins - 1)
    return (
        pd.DataFrame({"bin": b, "predicted": proba[:, 0], "observed": observed})
        .groupby("bin")
        .agg(
            layers=("predicted", "size"),
            predicted=("predicted", "mean"),
            observed=("observed", "mean"),
        )
        .round(2)
    )


def run(db_path: Path, out_dir: Path, fig_dir: Path, run_id: str = "em1dfm_pt1") -> dict:
    from morrill.analysis import figures

    con = duckdb.connect(str(db_path), read_only=True)
    pairs = training_pairs(con, run_id)
    profiles = load_profiles(con, run_id)
    depth_m = layer_mid_depths(con)
    metres = pairs[list(CLASSES)].to_numpy()
    print(
        f"training: {pairs['hole_id'].nunique()} holes, {len(pairs)} hole-layers, "
        + ", ".join(f"{c} {m:.0f} m" for c, m in zip(CLASSES, metres.sum(axis=0), strict=True))
    )

    variants = {
        "depth prior alone": {"use_resistivity": False},
        "histogram, overall prior": {"by_depth": False, "superposition": False},
        "histogram, depth prior": {"superposition": False},
        "histogram, depth prior + superposition": {},
    }
    cv = {name: leave_one_hole_out(pairs, profiles, depth_m, **kw) for name, kw in variants.items()}
    table = pd.DataFrame({name: scores(pairs, p) for name, p in cv.items()}).T
    print("\nleave-one-hole-out:")
    print(table.round(2).to_string())
    print(f"20 ohm·m cutoff, aquifer accuracy: {cutoff_baseline(pairs):.2f}")
    best = "histogram, depth prior + superposition"
    print(f"\ncalibration of P(coarse), {best}:")
    print(calibration(pairs, cv[best]).to_string())

    model = HistogramModel().fit(pairs)
    proba = model.predict_profiles(profiles.to_numpy(), depth_m)
    meta = con.sql("SELECT sounding_id, line, x, y FROM sounding").df().set_index("sounding_id")
    layers = con.sql("SELECT layer, top_m, bottom_m FROM model_layer").df().set_index("layer")
    n, k = profiles.shape
    cells = pd.DataFrame(
        {
            "sounding_id": np.repeat(profiles.index.to_numpy(), k),
            "layer": np.tile(profiles.columns.to_numpy(), n),
            "log_rho": profiles.to_numpy().ravel(),
        }
    )
    cells = cells.join(meta, on="sounding_id").join(layers, on="layer")
    for i, c in enumerate(CLASSES):
        cells[f"p_{c}"] = proba[:, :, i].ravel()
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out_dir / "lithology_probability.parquet")

    log_hist = con.sql(
        """
        SELECT log10(least(greatest(l.resistivity_ohm_m, 1), 1000)) AS log_rho, c.texture
        FROM induction_log_clean l
        JOIN lithology_classified c
          ON l.hole_id = c.hole_id AND l.depth_m >= c.from_m AND l.depth_m < c.to_m
        WHERE l.reliable
        """
    ).df()
    log_hist["cls"] = log_hist["texture"].map(CLASS_OF_TEXTURE)
    lith = con.sql("SELECT hole_id, from_m, to_m, texture FROM lithology_classified").df()
    lith["cls"] = lith["texture"].map(CLASS_OF_TEXTURE)
    holes = con.sql("SELECT hole_id, x, y FROM borehole WHERE has_lithology").df()

    fig_dir.mkdir(parents=True, exist_ok=True)
    figures.lithology_histograms(
        model, log_hist.dropna(subset=["cls"]), fig_dir / "lithology_histograms.png"
    )
    line = figures.lithology_section(cells, lith, holes, fig_dir / "lithology_section.png")
    print(f"\nsection drawn along line {line}")
    return {"scores": table, "model": model}
