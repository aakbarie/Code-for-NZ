"""Static figures for the learning notes (PNG, light surface).

Colours follow the project's chart rules: one highlight hue against grey for
small multiples (a map can't carry ten categorical colours legibly), and a
single-hue light-to-dark ramp for continuous depth.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
