"""Group soundings by the shape of their resistivity-depth profile.

UMAP's main product is a weighted nearest-neighbour graph: each sounding is
linked to the soundings whose profiles look most like it. Leiden finds
communities in that graph by modularity. The 2-D UMAP layout is used only
for pictures.

Why not cluster the 2-D picture (e.g. HDBSCAN)? The profiles form a
continuum, mostly in how deep the resistive sediments reach, and density
clustering cuts a continuum at arbitrary places. On this data HDBSCAN gave
13-45 groups depending on settings, with seed-to-seed agreement (adjusted
Rand index) of 0.4-0.76. Leiden on the graph at resolution 0.05 gave 10
communities for every seed, ARI 0.72-0.79, and the longest unbroken
stretches along the flight lines.
"""

from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROFILE_LAYERS = 21  # layers 1-21 cover the top 83.7 m; deeper layers are poorly resolved
RES_RANGE = (1.0, 1000.0)  # clip implausible values (0.35% of them) before taking logs


def load_profiles(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    """log10 resistivity, one row per sounding, one column per layer."""
    lo, hi = RES_RANGE
    df = con.execute(
        """
        SELECT sounding_id, layer, log10(least(greatest(resistivity_ohm_m, ?), ?)) AS lr
        FROM model_resistivity
        WHERE run_id = ? AND layer <= ? AND sounding_id IS NOT NULL
        """,
        [lo, hi, run_id, PROFILE_LAYERS],
    ).df()
    return df.pivot(index="sounding_id", columns="layer", values="lr").sort_index()


def neighbour_graph(X: np.ndarray, seed: int, n_neighbors: int = 30):
    """Return UMAP's weighted k-NN graph (sparse) and a 2-D layout for plotting."""
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.0, n_components=2, random_state=seed)
    layout = reducer.fit_transform(X)
    return reducer.graph_, layout


def leiden(graph, resolution: float, seed: int) -> np.ndarray:
    import igraph as ig
    import leidenalg as la

    g = graph.tocoo()
    G = ig.Graph(
        n=g.shape[0],
        edges=list(zip(g.row.tolist(), g.col.tolist(), strict=True)),
        edge_attrs={"weight": g.data.tolist()},
    ).simplify(combine_edges="max")
    part = la.find_partition(
        G,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.array(part.membership)


def order_communities(labels: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Renumber communities 1..k by the median of `key`, ascending; all-NaN groups last.

    Leiden's numbers are arbitrary; ordering by, say, depth to conductor makes
    community 1 mean the same thing from run to run.
    """
    med = pd.Series(key).groupby(labels).median().sort_values(kind="stable", na_position="last")
    mapping = {old: new for new, old in enumerate(med.index, start=1)}
    return np.array([mapping[v] for v in labels])


def summarize(con: duckdb.DuckDBPyConnection, out: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    layers = con.sql("SELECT layer, top_m, bottom_m FROM model_layer").df().set_index("layer")
    shallow = [c for c in X.columns if layers.loc[c, "bottom_m"] <= 10]
    mid = [c for c in X.columns if layers.loc[c, "top_m"] >= 20 and layers.loc[c, "bottom_m"] <= 40]
    g = out.groupby("community")
    lab = out["community"].to_numpy()
    # unbroken stretches along the flight lines (soundings are in flight order)
    starts = out["community"].ne(out["community"].shift()) | out["line"].ne(out["line"].shift())
    stretches = out.loc[starts, "community"].value_counts()
    summary = pd.DataFrame(
        {
            "soundings": g.size(),
            "stretches": stretches,
            "conductor_top_m": g["conductor_top_m"].median().round(1),
            "rho_0_10m": (10 ** X[shallow].mean(axis=1).groupby(lab).median()).round(0),
            "rho_20_40m": (10 ** X[mid].mean(axis=1).groupby(lab).median()).round(0),
            "bird_height_m": g["bird_height_m"].median().round(0),
        }
    )
    con.register("_comm", out.reset_index()[["sounding_id", "community"]])
    holes = con.sql(
        """
        SELECT c.community,
               string_agg(a.hole_id || ' (' || round(a.bedrock_depth_m, 1) || ' m)', ', '
                          ORDER BY a.bedrock_depth_m) AS boreholes_bedrock
        FROM (SELECT hole_id, sounding_id FROM borehole_sounding
              WHERE distance_m <= 100 AND sounding_id IN (SELECT sounding_id FROM _comm)
              QUALIFY row_number() OVER (PARTITION BY hole_id ORDER BY distance_m) = 1) AS bs
        JOIN _comm AS c USING (sounding_id)
        JOIN aquifer_base AS a USING (hole_id)
        WHERE a.reached_bedrock
        GROUP BY c.community
        """
    ).df()
    con.unregister("_comm")
    return summary.join(holes.set_index("community")).fillna({"boreholes_bedrock": ""})


def run(
    db_path: Path,
    out_dir: Path,
    fig_dir: Path,
    run_id: str = "em1dfm_pt1",
    seeds: tuple[int, ...] = (0, 1, 2),
    resolution: float = 0.05,
) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score

    from morrill.analysis import figures

    con = duckdb.connect(str(db_path), read_only=True)
    X = load_profiles(con, run_id)
    depth = (
        con.execute(
            "SELECT sounding_id, conductor_top_m FROM model_conductor_depth WHERE run_id = ?",
            [run_id],
        )
        .df()
        .set_index("sounding_id")["conductor_top_m"]
        .reindex(X.index)
    )

    runs = []
    for seed in seeds:
        graph, layout = neighbour_graph(X.to_numpy(), seed)
        runs.append((order_communities(leiden(graph, resolution, seed), depth.to_numpy()), layout))
    ari = [
        adjusted_rand_score(runs[i][0], runs[j][0]) for i, j in combinations(range(len(runs)), 2)
    ]

    labels, layout = runs[0]
    meta = (
        con.sql("SELECT sounding_id, line, x, y, bird_height_m FROM sounding")
        .df()
        .set_index("sounding_id")
    )
    out = meta.reindex(X.index).assign(
        community=labels, umap_x=layout[:, 0], umap_y=layout[:, 1], conductor_top_m=depth
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "communities.parquet")

    summary = summarize(con, out, X)
    print(f"{X.shape[0]:,} profiles, {summary.shape[0]} communities")
    print(f"seed-to-seed agreement (adjusted Rand): {np.round(ari, 2).tolist()}")
    print(summary.to_string())

    fig_dir.mkdir(parents=True, exist_ok=True)
    layers = con.sql("SELECT layer, top_m, bottom_m FROM model_layer ORDER BY layer").df()
    everything = con.sql("SELECT x, y FROM sounding").df()
    holes = con.sql(
        "SELECT hole_id, x, y, bedrock_depth_m FROM aquifer_base WHERE reached_bedrock"
    ).df()
    figures.communities(out, X, layers, summary, fig_dir / "communities.png")
    figures.conductor_depth(out, everything, holes, fig_dir / "conductor_depth.png")
    return summary
