"""From the depth to bedrock to the amount of aquifer material, with its uncertainty.

The sharp-contact inversion (bedrock.py) leaves 400 coherent surfaces per line:
the smoother updates whole members, so member i is one plausible bedrock surface
along the line. Each surface gives a thickness of aquifer material from the
ground down to bedrock, and integrated over the strip of ground each line
stands for, a volume. Comparing that with the volume implied by the smooth
model and by EM1DFM shows how much the choice of inversion changes the answer.

Saturated thickness, storage and transmissivity need water levels, specific
yield and hydraulic conductivity, which are not in the database; this stops at
the thickness of material between the ground and bedrock.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from morrill.analysis import bedrock, enki

NS_LINES = tuple(range(30010, 30101, 10))  # the tie lines cross these, so they're left out
MODEL_BOTTOM_M = 125.0  # "no conductor" counts as the bottom of the models


def strip_weights(y: np.ndarray) -> np.ndarray:
    """Metres of line each step stands for: half the gap to each neighbour (y sorted)."""
    gaps = np.diff(y)
    return np.r_[gaps[0], gaps[:-1] + gaps[1:], gaps[-1]] / 2


def strip_widths(easting: pd.Series) -> pd.Series:
    """Width of ground each line stands for: half the gap to each neighbouring line, so
    the strips cover the ground between the outermost lines and nothing beyond."""
    e = easting.sort_values()
    gaps = np.diff(e.to_numpy())
    return pd.Series(np.r_[gaps[0], gaps[:-1] + gaps[1:], gaps[-1]] / 2, index=e.index)


def recalibrate(members: np.ndarray, k: float) -> np.ndarray:
    """Each surface stretched k-fold about the per-step median, in log depth (as
    bedrock.widen does for the interval), and kept within the model's 1-150 m."""
    lm = np.log10(members)
    mid = np.median(lm, axis=1, keepdims=True)
    return np.clip(10.0 ** (mid + k * (lm - mid)), *10.0 ** np.array(bedrock.LOG_DEPTH_RANGE))


def _em1dfm(con, sounding_id: pd.Series) -> pd.Series:
    em = con.execute(
        "SELECT sounding_id, conductor_top_m FROM model_conductor_depth WHERE run_id = ?",
        [enki.INVERTED_RUN],
    ).df()
    on = sounding_id.map(em.set_index("sounding_id")["conductor_top_m"])
    inverted = sounding_id.isin(em["sounding_id"])
    return on.where(~(inverted & on.isna()), MODEL_BOTTOM_M)  # inverted, but no conductor


def thickness_table(steps, members, smooth, em1dfm, k) -> tuple[pd.DataFrame, dict]:
    """Per line: length-weighted mean thickness by each model; the sharp ensemble's
    realizations, raw and recalibrated, as 10/50/90% of the line mean."""
    rows, draws = [], {"raw": {}, "recalibrated": {}}
    for line in NS_LINES:
        sel = np.flatnonzero(steps["line"].to_numpy() == line)
        order = sel[np.argsort(steps["y"].to_numpy()[sel])]
        w = strip_weights(steps["y"].to_numpy()[order])

        def mean(d, w=w):
            ok = np.isfinite(d)  # EM1DFM skipped some soundings; average over the rest
            return float(w[ok] @ d[ok] / w[ok].sum()) if w[ok].sum() > 0.9 * w.sum() else np.nan

        raw = w @ members[order] / w.sum()
        cal = w @ recalibrate(members[order], k) / w.sum()
        draws["raw"][line], draws["recalibrated"][line] = raw, cal
        em = em1dfm[order]
        qs = (10, 50, 90)
        rows.append({
            "line": line, "length_km": w.sum() / 1000,
            "sharp_median": mean(steps["depth_q50"].to_numpy()[order]),
            **dict(zip([f"sharp_raw_q{q}" for q in qs], np.percentile(raw, qs), strict=True)),
            **dict(zip([f"sharp_cal_q{q}" for q in qs], np.percentile(cal, qs), strict=True)),
            "smooth": mean(np.minimum(smooth[order], MODEL_BOTTOM_M)),
            "em1dfm": mean(em),
        })  # fmt: skip
    return pd.DataFrame(rows), draws


def volumes(table: pd.DataFrame, draws: dict, widths: pd.Series) -> pd.DataFrame:
    """km³ of material between the ground and bedrock under the ten lines' strips.
    Lines are inverted independently, so member i of one line pairs freely with
    member i of another."""
    area = table.set_index("line")["length_km"] * 1000 * widths.reindex(table["line"]).to_numpy()
    out = {}
    for name in ("raw", "recalibrated"):
        total = sum(area[line] * draws[name][line] for line in NS_LINES) / 1e9
        out[f"sharp, {name}"] = np.quantile(total, [0.1, 0.5, 0.9])
    for col in ("sharp_median", "smooth", "em1dfm"):
        v = (area * table.set_index("line")[col]).sum(min_count=len(NS_LINES)) / 1e9
        out[col] = [np.nan, v, np.nan]
    six = [L for L in NS_LINES if np.isfinite(table.set_index("line").loc[L, "em1dfm"])]
    for col in ("sharp_median", "smooth", "em1dfm"):
        out[f"{col}, EM1DFM's {len(six)} lines"] = [
            np.nan,
            (area[six] * table.set_index("line").loc[six, col]).sum() / 1e9,
            np.nan,
        ]
    return pd.DataFrame(out, index=["q10", "q50", "q90"]).T


def run(db_path: Path, out_dir: Path, fig_dir: Path) -> dict:
    from morrill.analysis import figures

    con = duckdb.connect(str(db_path), read_only=True)
    steps = pd.read_parquet(out_dir / "bedrock_soundings.parquet")
    saved = np.load(out_dir / "bedrock_members.npz")
    assert np.array_equal(saved["sounding_id"], steps["sounding_id"].to_numpy())
    calib = pd.read_parquet(out_dir / "bedrock_calibration.parquet")
    k = bedrock.inflation(calib["score"].to_numpy())
    smooth = (
        pd.read_parquet(out_dir / "enki_soundings.parquet")
        .set_index("sounding_id")["depth_q50"]
        .reindex(steps["sounding_id"])
        .to_numpy()
    )
    em1dfm = _em1dfm(con, steps["sounding_id"]).to_numpy()
    widths = strip_widths(steps[steps["line"].isin(NS_LINES)].groupby("line")["x"].median())

    table, draws = thickness_table(steps, saved["depth_m"], smooth, em1dfm, k)
    vol = volumes(table, draws, widths)
    print(f"Mean thickness from the ground to bedrock, by line (m); recalibration k = {k:.2f}:")
    print(table.round(1).to_string(index=False))
    print("\nVolume of aquifer material under the ten lines' strips (km³):")
    print(vol.round(3).to_string())
    table.to_parquet(out_dir / "aquifer_thickness.parquet")
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures.aquifer_thickness(table, fig_dir / "aquifer_thickness.png")
    return {"table": table, "volumes": vol, "k": k}
