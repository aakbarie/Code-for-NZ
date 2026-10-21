"""Static figures for the learning notes (PNG, light surface).

Colours follow the project's chart rules: one highlight hue against grey for
small multiples (a map can't carry ten categorical colours legibly), and a
single-hue light-to-dark ramp for continuous depth.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
HIGHLIGHT = "#2a78d6"
CONTEXT = "#d3d1cb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": CONTEXT,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 8,
        "axes.titlesize": 9,
    }
)


def _map_axes(ax, xy: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(xy["x"].min() / 1000 - 0.3, xy["x"].max() / 1000 + 0.3)
    ax.set_ylim(xy["y"].min() / 1000 - 0.3, xy["y"].max() / 1000 + 0.3)
    ax.tick_params(length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def communities(
    out: pd.DataFrame,
    X: pd.DataFrame,
    layers: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """One column per community: where it is (top) and its typical profile (bottom)."""
    ks = list(summary.index)
    fig, axes = plt.subplots(
        2, len(ks), figsize=(1.45 * len(ks), 12.5), gridspec_kw={"height_ratios": [4, 1.3]}
    )
    mids = layers.set_index("layer").loc[X.columns]
    depth = (mids["top_m"] + mids["bottom_m"]) / 2
    for col, k in enumerate(ks):
        ax = axes[0, col]
        sel = out["community"] == k
        ax.scatter(out["x"] / 1000, out["y"] / 1000, s=0.6, c=CONTEXT, lw=0, rasterized=True)
        ax.scatter(
            out.loc[sel, "x"] / 1000, out.loc[sel, "y"] / 1000, s=0.8, c=HIGHLIGHT, lw=0,
            rasterized=True,
        )  # fmt: skip
        _map_axes(ax, out)
        if col:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("northing (km, UTM 13N)")
        ax.set_xticks([])
        row = summary.loc[k]
        depth_txt = "none" if pd.isna(row["conductor_top_m"]) else f"{row['conductor_top_m']:g} m"
        ax.set_title(f"{k}\n{int(row['soundings']):,} soundings\nconductor {depth_txt}", loc="left")

        ax = axes[1, col]
        rho = 10 ** X[sel.to_numpy()]
        q25, q50, q75 = (rho.quantile(q) for q in (0.25, 0.5, 0.75))
        ax.fill_betweenx(depth, q25, q75, color=HIGHLIGHT, alpha=0.2, lw=0)
        ax.plot(q50, depth, color=HIGHLIGHT, lw=2)
        ax.axvline(20, color=INK_2, lw=0.8, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_xlim(2, 1000)
        ax.set_ylim(depth.max() + 5, 0)
        ax.grid(axis="x", color=CONTEXT, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if col:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("depth (m)")
        ax.set_xlabel("ohm·m")
    fig.suptitle(
        "Communities of similar resistivity profiles (UMAP neighbour graph + Leiden), "
        "EM1DFM inversion, Morrill block\n"
        "Top: where each community is (blue) among all inverted soundings. "
        "Bottom: median profile, band = middle 50%, dashed = 20 ohm·m.",
        x=0.01, ha="left", fontsize=10,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def conductor_depth(
    out: pd.DataFrame, everything: pd.DataFrame, holes: pd.DataFrame, path: Path
) -> None:
    """Depth to the first conductive layer, with borehole bedrock depths on the same scale."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.6, 13))
    vmax = 80
    ax.scatter(
        everything["x"] / 1000, everything["y"] / 1000, s=1, c=CONTEXT, lw=0, rasterized=True
    )
    has = out["conductor_top_m"].notna()
    sc = ax.scatter(
        out.loc[has, "x"] / 1000, out.loc[has, "y"] / 1000, s=4,
        c=out.loc[has, "conductor_top_m"].clip(upper=vmax), cmap=BLUE_RAMP, vmin=0, vmax=vmax,
        lw=0, rasterized=True,
    )  # fmt: skip
    ax.scatter(
        out.loc[~has, "x"] / 1000, out.loc[~has, "y"] / 1000, s=4, c=INK_2, lw=0,
        rasterized=True,
    )  # fmt: skip
    inside = holes[
        holes["x"].between(everything["x"].min() - 300, everything["x"].max() + 300)
        & holes["y"].between(everything["y"].min() - 300, everything["y"].max() + 300)
    ]
    ax.scatter(
        inside["x"] / 1000, inside["y"] / 1000, s=42, c=inside["bedrock_depth_m"].clip(upper=vmax),
        cmap=BLUE_RAMP, vmin=0, vmax=vmax, edgecolors=INK, linewidths=1.2, zorder=3,
    )  # fmt: skip
    _map_axes(ax, everything)
    ax.set_xlabel("easting (km, UTM 13N)")
    ax.set_ylabel("northing (km, UTM 13N)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.35, pad=0.02, extend="max")
    cb.set_label("depth (m)")
    cb.ax.invert_yaxis()

    def dot(color, size, edge="none"):
        return Line2D([], [], ls="", marker="o", ms=size, mfc=color, mec=edge)

    ax.legend(
        [dot(CONTEXT, 4), dot(INK_2, 4), dot("#86b6ef", 7, INK)],
        ["not inverted", "no conductor in top 125 m", "borehole: depth to bedrock"],
        loc="lower left", fontsize=7, frameon=False,
    )  # fmt: skip
    ax.set_title(
        "Depth to the first layer below 20 ohm·m (inversion)\n"
        "vs depth to bedrock in boreholes (ringed)",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Rock classes: categorical slots 1-3 (the three that stay distinguishable in any pairing).
CLASS_COLOURS = {"coarse": "#2a78d6", "fine": "#eb6834", "bedrock": "#1baf7a"}
CLASS_LABELS = {
    "coarse": "coarse sediment (sand, gravel)",
    "fine": "fine sediment (silt, clay)",
    "bedrock": "bedrock (siltstone, Brule)",
}


def _rho_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_xlabel("resistivity (ohm·m)")
    ax.grid(axis="x", color=CONTEXT, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def lithology_histograms(model, log_hist: pd.DataFrame, path: Path) -> None:
    """Class likelihoods from the inversion, the same from the logs, and posteriors by depth."""
    from morrill.analysis.lithology import BINS, CLASSES, DEPTH_BANDS

    centres = 10 ** ((BINS[:-1] + BINS[1:]) / 2)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)

    ax = axes[0]
    for i, c in enumerate(CLASSES):
        ax.plot(centres, model.likelihood_[i], color=CLASS_COLOURS[c], lw=2, label=CLASS_LABELS[c])
        peak = model.likelihood_[i].argmax()
        ax.annotate(c, (centres[peak], model.likelihood_[i][peak]), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK_2, fontsize=7)  # fmt: skip
    ax.set_title("P(resistivity | class)\nfrom the INVERSION at boreholes", loc="left")
    ax.set_ylabel("share per 0.1-decade bin")

    ax = axes[1]
    for c in CLASSES:
        v = log_hist.loc[log_hist["cls"] == c, "log_rho"].to_numpy()
        h, _ = np.histogram(v, bins=BINS)
        ax.plot(centres, h / max(h.sum(), 1), color=CLASS_COLOURS[c], lw=2)
    ax.set_title("the same, from the borehole LOGS\n(3 cm readings)", loc="left")

    for ax, (band, label) in zip(axes[2:], [(1, "at 5-15 m"), (3, "below 30 m")], strict=True):
        depth = np.full(len(centres), (DEPTH_BANDS[band] + min(DEPTH_BANDS[band + 1], 84)) / 2)
        proba = model.predict_cells(np.log10(centres), depth)
        for i, c in enumerate(CLASSES):
            ax.plot(centres, proba[:, i], color=CLASS_COLOURS[c], lw=2)
        ax.axvline(20, color=INK_2, lw=0.8, ls=(0, (2, 2)))
        ax.set_ylim(0, 1)
        ax.set_title(
            f"P(class | resistivity) {label}\n(dashed: the 2012 20 ohm·m cutoff)", loc="left"
        )
    axes[2].set_ylabel("probability")
    for ax in axes:
        _rho_axis(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def lithology_section(
    cells: pd.DataFrame, lith: pd.DataFrame, holes: pd.DataFrame, path: Path, max_dist_m=200.0
) -> int:
    """Most likely class along the flight line with the most nearby boreholes.

    Returns the line number drawn.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from scipy.spatial import cKDTree

    from morrill.analysis.lithology import CLASSES

    soundings = cells.drop_duplicates("sounding_id")[["sounding_id", "line", "x", "y"]]
    dist, idx = cKDTree(soundings[["x", "y"]].to_numpy()).query(holes[["x", "y"]].to_numpy())
    holes = holes.assign(line=soundings["line"].to_numpy()[idx], dist=dist)
    holes = holes[holes["dist"] <= max_dist_m]
    line = int(holes["line"].value_counts().idxmax())

    sec = cells[cells["line"] == line].sort_values(["y", "layer"])
    ys = sec.drop_duplicates("sounding_id")["y"].to_numpy() / 1000
    n_layers = sec["layer"].max()
    p = sec[[f"p_{c}" for c in CLASSES]].to_numpy().reshape(len(ys), n_layers, len(CLASSES))
    best, conf = p.argmax(axis=2).T, p.max(axis=2).T
    tops = sec.drop_duplicates("layer").sort_values("layer")
    yedges = np.r_[tops["top_m"].to_numpy(), tops["bottom_m"].to_numpy()[-1]]
    xedges = np.r_[ys[0], (ys[1:] + ys[:-1]) / 2, ys[-1]]

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]}
    )
    ax = axes[0]
    ax.pcolormesh(xedges, yedges, best, cmap=ListedColormap([CLASS_COLOURS[c] for c in CLASSES]),
                  vmin=-0.5, vmax=2.5, rasterized=True)  # fmt: skip
    on_line = holes[holes["line"] == line]
    for _, h in on_line.iterrows():
        for _, r in lith[lith["hole_id"] == h["hole_id"]].iterrows():
            if r["from_m"] >= yedges[-1]:
                continue
            colour = CLASS_COLOURS.get(r["cls"], CONTEXT)
            ax.add_patch(Rectangle((h["y"] / 1000 - 0.06, r["from_m"]), 0.12,
                                   min(r["to_m"], yedges[-1]) - r["from_m"],
                                   facecolor=colour, edgecolor=INK, lw=0.6, zorder=3))  # fmt: skip
        ax.annotate(h["hole_id"], (h["y"] / 1000, 0), xytext=(0, 3), textcoords="offset points",
                    rotation=90, ha="center", va="bottom", fontsize=6, color=INK_2)  # fmt: skip
    ax.set_ylim(yedges[-1], 0)
    ax.set_ylabel("depth (m)")
    ax.set_title(
        f"Line {line}: most likely class from the histogram model; boreholes within "
        f"{max_dist_m:.0f} m drawn as logged (grey = not modelled: soil, sandstone, ...)",
        loc="left", pad=28,
    )  # fmt: skip
    handles = [Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[c]) for c in CLASSES]
    ax.legend(handles, [CLASS_LABELS[c] for c in CLASSES], loc="lower left", ncol=3, frameon=False,
              bbox_to_anchor=(0, 1.0), fontsize=7)  # fmt: skip

    ax = axes[1]
    grey = LinearSegmentedColormap.from_list("grey_seq", ["#f0efec", "#52514e"])
    mesh = ax.pcolormesh(xedges, yedges, conf, cmap=grey, vmin=1 / 3, vmax=1, rasterized=True)
    ax.set_ylim(yedges[-1], 0)
    ax.set_ylabel("depth (m)")
    ax.set_xlabel("northing (km, UTM 13N)")
    ax.set_title("confidence: probability of the most likely class", loc="left")
    fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03)
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return line


def kalman_depth_map(
    everything: pd.DataFrame, depth: pd.Series, holes: pd.DataFrame, path: Path
) -> None:
    """Depth to conductor predicted from Kalman features on all lines, boreholes ringed."""
    from matplotlib.lines import Line2D

    vmax = 80
    pts = everything.set_index("sounding_id").join(depth.rename("depth"))
    fig, ax = plt.subplots(figsize=(5.6, 13))
    has = pts["depth"].notna()
    sc = ax.scatter(
        pts.loc[has, "x"] / 1000, pts.loc[has, "y"] / 1000, s=4,
        c=pts.loc[has, "depth"].clip(0, vmax), cmap=BLUE_RAMP, vmin=0, vmax=vmax, lw=0,
        rasterized=True,
    )  # fmt: skip
    inside = holes[
        holes["x"].between(pts["x"].min() - 300, pts["x"].max() + 300)
        & holes["y"].between(pts["y"].min() - 300, pts["y"].max() + 300)
    ]
    ax.scatter(
        inside["x"] / 1000, inside["y"] / 1000, s=42, c=inside["bedrock_depth_m"].clip(upper=vmax),
        cmap=BLUE_RAMP, vmin=0, vmax=vmax, edgecolors=INK, linewidths=1.2, zorder=3,
    )  # fmt: skip
    _map_axes(ax, pts.reset_index())
    ax.set_xlabel("easting (km, UTM 13N)")
    ax.set_ylabel("northing (km, UTM 13N)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.35, pad=0.02, extend="max")
    cb.set_label("depth (m)")
    cb.ax.invert_yaxis()
    handle = Line2D([], [], ls="", marker="o", ms=7, mfc="#86b6ef", mec=INK)
    ax.legend([handle], ["borehole: depth to bedrock"], loc="lower left", fontsize=7, frameon=False)
    ax.set_title(
        "Depth to conductor predicted from Kalman features, all 10 lines\n"
        "(inverted lines: each predicted with its own line held out)",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def kalman_line(
    everything: pd.DataFrame,
    feats: pd.DataFrame,
    pred: pd.DataFrame,
    holes: pd.DataFrame,
    line: int,
    path: Path,
) -> None:
    """One flight line: bird height, the Kalman features, surprise, and predicted depth."""
    sel = everything[everything["line"] == line].set_index("sounding_id")
    sel = sel.join(feats).join(pred)
    y = sel["y"].to_numpy() / 1000
    zs = [c for c in feats.columns if c.startswith("z")]
    n_panels = 3 + len(zs) + 1
    ratios = [1] + [1] * len(zs) + [1, 1, 2.2]
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(13, 2.0 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": ratios},
    )
    axes[0].plot(y, sel["bird_height_m"], color=INK_2, lw=1)
    axes[0].set_ylabel("bird height (m)")
    for ax, z in zip(axes[1 : 1 + len(zs)], zs, strict=True):
        sd = sel[z.replace("z", "sd")]
        ax.fill_between(y, sel[z] - 2 * sd, sel[z] + 2 * sd, color=HIGHLIGHT, alpha=0.2, lw=0)
        ax.plot(y, sel[z], color=HIGHLIGHT, lw=1.2)
        ax.set_ylabel(f"feature {z}")
    ax = axes[1 + len(zs)]
    ax.plot(y, sel["nis"], color=INK, lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("innovation\nscore")
    ax = axes[2 + len(zs)]
    ax.plot(y, sel["powerline_monitor"], color=INK_2, lw=0.8)
    ax.set_ylabel("power-line\nmonitor")
    ax = axes[-1]
    ax.fill_between(
        y, sel["q10"], sel["q90"], color=HIGHLIGHT, alpha=0.2, lw=0, label="80% interval"
    )
    ax.plot(y, sel["q50"], color=HIGHLIGHT, lw=1.5, label="predicted depth to conductor")
    on = holes[holes["line"] == line]
    hy = everything.set_index("sounding_id").loc[on["sounding_id"], "y"].to_numpy() / 1000
    ax.scatter(
        hy, on["bedrock_depth_m"], s=40, color=SURFACE, edgecolors=INK, linewidths=1.2,
        zorder=3, label="borehole: depth to bedrock",
    )  # fmt: skip
    for yy, (_, r) in zip(hy, on.iterrows(), strict=True):
        ax.annotate(
            r["hole_id"], (yy, r["bedrock_depth_m"]), xytext=(4, 4),
            textcoords="offset points", fontsize=6, color=INK_2,
        )  # fmt: skip
    ax.invert_yaxis()
    ax.set_ylabel("depth (m)")
    ax.set_xlabel("northing (km, UTM 13N)")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    for a in axes:
        a.grid(axis="x", color=CONTEXT, lw=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle(
        f"Line {line} (never inverted): Kalman features from the raw channels, and the "
        "depth to conductor they predict", x=0.01, ha="left", fontsize=10,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- ensemble Kalman inversion -------------------------------------------------------------

ORANGE_RAMP = LinearSegmentedColormap.from_list(
    "orange_seq", ["#fde5d9", "#f8b897", "#f28b5e", "#eb6834", "#c24e1f", "#8f3714", "#5f240c"]
)
RHO_TICKS = [3, 10, 30, 100, 300]
RHO_LIMITS = (0.5, 2.5)  # log10 ohm·m


def _wide(layer_table: pd.DataFrame, ids, col: str):
    return layer_table.pivot(index="sounding_id", columns="layer", values=col).loc[ids].to_numpy()


def _edges(v):
    mid = (v[1:] + v[:-1]) / 2
    return np.r_[v[0] - (mid[0] - v[0]), mid, v[-1] + (v[-1] - mid[-1])]


def _section(ax, fig, along_km, grid, lay, cmap, vmin, vmax, label, rho=False):
    depth_edges = np.r_[lay["top_m"].to_numpy()[:-1], 125.0]  # half-space not drawn
    order = np.argsort(along_km)
    mesh = ax.pcolormesh(
        _edges(along_km[order]), depth_edges, grid[order][:, :-1].T, cmap=cmap,
        vmin=vmin, vmax=vmax, shading="flat", rasterized=True,
    )  # fmt: skip
    ax.set_ylim(125, 0)
    ax.set_ylabel("depth (m)")
    cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label(label)
    if rho:
        cb.set_ticks(np.log10(RHO_TICKS), labels=[str(t) for t in RHO_TICKS])
        cb.ax.invert_yaxis()
    return mesh


def _holes_on(ax, holes, line, along, label=True):
    on = holes[holes["line"] == line]
    for _, r in on.iterrows():
        yy = along(r)
        ax.plot([yy, yy], [0, r["bedrock_depth_m"]], color=INK, lw=0.8)
        ax.scatter(
            yy, r["bedrock_depth_m"], s=36, color=SURFACE, edgecolors=INK, linewidths=1.2, zorder=4
        )
        if label:
            ax.annotate(
                r["hole_id"], (yy, 0), xytext=(3, 3), textcoords="offset points",
                fontsize=6, color=INK_2, va="bottom",
            )  # fmt: skip


def _tidy(axes):
    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)


def enki_section(steps, layer_table, lay, holes, line, path: Path) -> None:
    """One line: median resistivity, its uncertainty, depth to conductor, and the data fit."""
    sel = steps[steps["line"] == line].reset_index(drop=True)
    y = sel["y"].to_numpy() / 1000
    ids = sel["sounding_id"].to_numpy()
    pos = dict(zip(ids, y, strict=True))
    along = lambda r: pos[r["sounding_id"]]  # noqa: E731
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [2.2, 2.2, 2.2, 1]}
    )
    q50 = _wide(layer_table, ids, "q50")
    width = _wide(layer_table, ids, "q90") - _wide(layer_table, ids, "q10")
    _section(
        axes[0], fig, y, q50, lay, BLUE_RAMP.reversed(), *RHO_LIMITS,
        "median resistivity (ohm·m)\ndarker = more conductive", rho=True,
    )  # fmt: skip
    axes[0].plot(y, np.where(np.isinf(sel["depth_q50"]), np.nan, sel["depth_q50"]), color=SURFACE,
                 lw=1.2, label="depth to conductor (median)")  # fmt: skip
    _holes_on(axes[0], holes, line, along)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False, labelcolor=INK)
    axes[0].set_title("Median of the ensemble", loc="left")
    _section(axes[1], fig, y, width, lay, ORANGE_RAMP, 0, 1.2, "80% interval width\n(decades)")
    _holes_on(axes[1], holes, line, along, label=False)
    axes[1].set_title(
        "Uncertainty: width of the 80% interval of log10 resistivity (1 = a factor of 10)",
        loc="left",
    )
    ax = axes[2]
    lo, hi = sel["depth_q10"].to_numpy(), np.minimum(sel["depth_q90"].to_numpy(), 125)
    ax.fill_between(y, lo, hi, color=HIGHLIGHT, alpha=0.25, lw=0, label="along-line EnKI, 80%")
    ax.plot(y, np.minimum(sel["depth_q50"], 125), color=HIGHLIGHT, lw=1.5, label="median")
    on = holes[holes["line"] == line]
    hy = np.array([along(r) for _, r in on.iterrows()])
    if len(on):
        ax.errorbar(
            hy + 0.02, np.minimum(on["independent_q50"], 125),
            yerr=[np.minimum(on["independent_q50"], 125) - on["independent_q10"],
                  np.minimum(on["independent_q90"], 125) - np.minimum(on["independent_q50"], 125)],
            fmt="s", ms=4, color=INK_2, lw=1, capsize=2, label="sounding on its own, 80%",
        )  # fmt: skip
        ax.scatter(
            hy, on["bedrock_depth_m"], s=40, color=SURFACE, edgecolors=INK, linewidths=1.2,
            zorder=4, label="borehole: depth to bedrock",
        )  # fmt: skip
    ax.set_ylim(125, 0)
    ax.set_ylabel("depth to conductor (m)")
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncols=2)
    ax.set_title("Depth to the first layer below 1.8 m under 20 ohm·m", loc="left")
    ax = axes[3]
    ax.plot(y, sel["misfit"], color=INK, lw=0.8)
    ax.axhline(1, color=CONTEXT, lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("data misfit\n(1 = to the error)")
    ax.set_xlabel("northing (km, UTM 13N)")
    _tidy(axes)
    fig.suptitle(
        f"Line {line}: ensemble Kalman inversion with a 1D EM forward model", x=0.01, ha="left",
        fontsize=10,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def enki_compare(steps, layer_table, em, lay, holes, line, path: Path) -> None:
    """An inverted line: EM1DFM next to the EnKI median and its uncertainty."""
    sel = steps[steps["line"] == line].reset_index(drop=True)
    ids = sel["sounding_id"].to_numpy()
    y = sel["y"].to_numpy() / 1000
    pos = dict(zip(ids, y, strict=True))
    along = lambda r: pos[r["sounding_id"]]  # noqa: E731
    em_grid = em.pivot(index="sounding_id", columns="layer", values="em1dfm").loc[ids].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    label = "resistivity (ohm·m)\ndarker = more conductive"
    _section(axes[0], fig, y, em_grid, lay, BLUE_RAMP.reversed(), *RHO_LIMITS, label, rho=True)
    axes[0].set_title("EM1DFM (deterministic, each sounding on its own)", loc="left")
    _section(
        axes[1], fig, y, _wide(layer_table, ids, "q50"), lay, BLUE_RAMP.reversed(), *RHO_LIMITS,
        label, rho=True,
    )  # fmt: skip
    axes[1].set_title("EnKI median (along-line ensemble Kalman inversion)", loc="left")
    width = _wide(layer_table, ids, "q90") - _wide(layer_table, ids, "q10")
    _section(axes[2], fig, y, width, lay, ORANGE_RAMP, 0, 1.2, "80% interval width\n(decades)")
    axes[2].set_title("EnKI uncertainty: width of the 80% interval", loc="left")
    for i, ax in enumerate(axes):
        _holes_on(ax, holes, line, along, label=i == 0)
    axes[-1].set_xlabel("northing (km, UTM 13N)")
    _tidy(axes)
    fig.suptitle(f"Line {line}: two inversions of the same data", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def enki_depth_map(steps, bedrock, path: Path) -> None:
    """Depth to conductor (median) and its 80% interval width, on every line."""
    from matplotlib.lines import Line2D

    vmax = 80
    fig, axes = plt.subplots(1, 2, figsize=(10, 12.5), sharey=True)
    depth = np.minimum(steps["depth_q50"], 125)
    width = np.minimum(steps["depth_q90"], 125) - steps["depth_q10"]
    for ax, values, cmap, top, label, title in (
        (axes[0], depth, BLUE_RAMP, vmax, "depth (m)", "Depth to conductor, median"),
        (axes[1], width, ORANGE_RAMP, 40, "80% interval width (m)", "Its uncertainty"),
    ):
        sc = ax.scatter(
            steps["x"] / 1000, steps["y"] / 1000, s=7, c=np.clip(values, 0, top), cmap=cmap,
            vmin=0, vmax=top, lw=0, rasterized=True,
        )  # fmt: skip
        _map_axes(ax, steps)
        ax.set_xlabel("easting (km, UTM 13N)")
        cb = fig.colorbar(sc, ax=ax, shrink=0.3, pad=0.02, extend="max")
        cb.set_label(label)
        if ax is axes[0]:
            cb.ax.invert_yaxis()
        ax.set_title(title, loc="left")
    inside = bedrock[
        bedrock["x"].between(steps["x"].min() - 300, steps["x"].max() + 300)
        & bedrock["y"].between(steps["y"].min() - 300, steps["y"].max() + 300)
    ]
    axes[0].scatter(
        inside["x"] / 1000, inside["y"] / 1000, s=42, c=inside["bedrock_depth_m"].clip(upper=vmax),
        cmap=BLUE_RAMP, vmin=0, vmax=vmax, edgecolors=INK, linewidths=1.2, zorder=3,
    )  # fmt: skip
    handle = Line2D([], [], ls="", marker="o", ms=7, mfc="#86b6ef", mec=INK)
    axes[0].legend(
        [handle], ["borehole: depth to bedrock"], loc="lower left", fontsize=7, frameon=False
    )
    axes[0].set_ylabel("northing (km, UTM 13N)")
    fig.suptitle(
        "Ensemble Kalman inversion of all 12 lines (every 20th sounding)", x=0.01, ha="left",
        fontsize=10,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)
