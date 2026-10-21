"""Features along flight lines from a Kalman state-space model of the raw HEM channels.

Each flight line is one sequence. The 12 channels (log ppm, standardised) are
modelled as a 4-number hidden state that drifts smoothly along the line, seen
through C, plus a bird-height term D·u (u = 1, log h, log² h). The smoothed
state, after a canonical rotation, gives the features z1..z4. Each sounding
also gets the features' uncertainty and an innovation score (how surprising
the sounding was given its predecessors).

Raw channels exist on all 12 lines, so the features do too, including the four
lines (L30070-L30100) and the tie lines that were never inverted.

Checks, printed by run():
1. Do the features still depend on bird height? (compared with raw channels)
2. Do innovation spikes coincide with the power-line monitor?
3. Leave-one-inverted-line-out: how well do the features predict the
   inversion's depth to conductor, compared with raw channels, raw channels
   smoothed along the line, and the features from a model without the height
   term?
4. Out of sample: predict depth to conductor on the uninverted lines and
   compare with borehole bedrock depths there.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from morrill.analysis import statespace as ss
from morrill.ingest.airborne import CHANNELS

K = 4
CHANNEL_COLS = list(CHANNELS.values())
INVERTED_RUN = "em1dfm_pt1"
SMOOTH_WINDOW = 31  # soundings (about 100 m), for the "just smooth it" baseline
MAX_HEIGHT_M = 60.0  # above this the bird barely sees the ground: treated as missing


def load_survey(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.sql(
        """
        SELECT s.sounding_id, s.line, s.is_tie_line, s.x, s.y, s.bird_height_m,
               s.powerline_monitor, o.* EXCLUDE (sounding_id)
        FROM sounding AS s JOIN hem_obs AS o USING (sounding_id)
        ORDER BY s.sounding_id
        """
    ).df()


def design(survey: pd.DataFrame, use_height: bool = True):
    """Per-line observations, inputs, observed masks, row indices, and the full Y."""
    logs = np.log(survey[CHANNEL_COLS].to_numpy())
    Y = (logs - logs.mean(axis=0)) / logs.std(axis=0)
    lh = np.log(survey["bird_height_m"].to_numpy())
    lh = lh - np.median(lh)
    U = np.c_[np.ones(len(lh)), lh, lh**2] if use_height else np.ones((len(lh), 1))
    observed = survey["bird_height_m"].to_numpy() <= MAX_HEIGHT_M
    lines = survey["line"].to_numpy()
    idx = [np.flatnonzero(lines == L) for L in pd.unique(lines)]
    return [Y[i] for i in idx], [U[i] for i in idx], [observed[i] for i in idx], idx, Y


def extract(survey: pd.DataFrame, k: int = K, use_height: bool = True, verbose: bool = False):
    """Fit the model on all lines and return per-sounding features."""
    Ys, Us, obs, idx, _ = design(survey, use_height)
    model, history = ss.em(Ys, Us, k=k, observed=obs, verbose=verbose)

    n = len(survey)
    xs_all, ps_all, nis = np.empty((n, k)), np.empty((n, k, k)), np.empty(n)
    for Y, U, o, i in zip(Ys, Us, obs, idx, strict=True):
        f = ss.kalman_filter(model, Y, U, o)
        xs, Ps, _ = ss.rts_smoother(model, f)
        xs_all[i], ps_all[i], nis[i] = xs, Ps, f.nis
    T = ss.canonical_rotation(model, xs_all)
    z = (xs_all - xs_all.mean(axis=0)) @ T.T
    sd = np.sqrt(np.einsum("ij,tjk,ik->ti", T, ps_all, T))
    feats = pd.DataFrame(
        {
            **{f"z{j + 1}": z[:, j] for j in range(k)},
            **{f"sd{j + 1}": sd[:, j] for j in range(k)},
            "nis": nis,
            "observed": survey["bird_height_m"].to_numpy() <= MAX_HEIGHT_M,
        },
        index=survey["sounding_id"],
    )
    return feats, model, history, T


def _smoothed(survey: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return survey.groupby("line")[cols].transform(
        lambda s: s.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    )


def feature_sets(survey: pd.DataFrame, feats: pd.DataFrame, feats_no_h: pd.DataFrame) -> dict:
    logs = np.log(survey[CHANNEL_COLS])
    logs.columns = [f"log_{c}" for c in CHANNEL_COLS]
    raw = logs.assign(log_height=np.log(survey["bird_height_m"]))
    raw.index = survey["sounding_id"]
    smooth = _smoothed(
        pd.concat([survey[["line"]], raw.reset_index(drop=True)], axis=1), list(raw.columns)
    )
    smooth.index = survey["sounding_id"]
    zs = [c for c in feats.columns if c.startswith("z")]
    return {
        "raw channels + height": raw,
        f"raw, smoothed over {SMOOTH_WINDOW}": smooth,
        "Kalman, no height term": feats_no_h[zs],
        "Kalman features": feats[zs],
    }


def leave_one_line_out(X: pd.DataFrame, target: pd.Series, lines: pd.Series, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    pred = pd.Series(np.nan, index=target.index)
    for L in sorted(lines.loc[target.index].unique()):
        test = (lines.loc[target.index] == L).to_numpy()
        model = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
        model.fit(X.loc[target.index[~test]], target[~test])
        pred[test] = model.predict(X.loc[target.index[test]])
    return pred


def fit_quantiles(X: pd.DataFrame, target: pd.Series, quantiles=(0.1, 0.5, 0.9), seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return {
        q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=300, random_state=seed
        ).fit(X.loc[target.index], target)
        for q in quantiles
    }


def run(db_path: Path, out_dir: Path, fig_dir: Path) -> dict:
    from morrill.analysis import figures

    con = duckdb.connect(str(db_path), read_only=True)
    survey = load_survey(con)
    print(f"{len(survey):,} soundings on {survey['line'].nunique()} lines; fitting k={K} ...")
    feats, model, history, _ = extract(survey)
    print(
        f"EM: {len(history)} iterations, log-likelihood {history[0]:,.0f} -> {history[-1]:,.0f}; "
        f"last step changed it by {(history[-1] - history[-2]) / abs(history[-2]):.1e} (relative)"
    )
    print(
        f"soundings above {MAX_HEIGHT_M:.0f} m treated as missing: {(~feats['observed']).sum():,}"
    )
    feats_no_h, _, _, _ = extract(survey, use_height=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.join(survey.set_index("sounding_id")[["line", "x", "y"]]).to_parquet(
        out_dir / "kalman_features.parquet"
    )

    # 1. bird height
    lh = np.log(survey["bird_height_m"].to_numpy())
    Ystd = design(survey)[-1]
    raw_corr = max(abs(spearmanr(Ystd[:, j], lh).statistic) for j in range(Ystd.shape[1]))
    z_corr = max(abs(spearmanr(feats[f"z{j + 1}"], lh).statistic) for j in range(K))
    zn_corr = max(abs(spearmanr(feats_no_h[f"z{j + 1}"], lh).statistic) for j in range(K))
    print("\n1. max |Spearman| with log bird height:")
    print(
        f"   raw channels {raw_corr:.2f} | Kalman, no height term {zn_corr:.2f}"
        f" | Kalman features {z_corr:.2f}"
    )

    # 2. power lines
    pl = survey["powerline_monitor"].to_numpy()
    top = feats["nis"].to_numpy() >= np.nanquantile(feats["nis"], 0.99)
    noisy = pl >= np.nanquantile(pl, 0.99)
    rho_pl = spearmanr(feats["nis"], pl, nan_policy="omit").statistic
    print("\n2. innovation score vs power-line monitor:")
    print(
        f"   Spearman {rho_pl:.2f}; of the 1% most surprising soundings, "
        f"{np.mean(noisy[top]):.0%} are in the top 1% of power-line noise (chance: 1%)"
    )

    # 3. predicting the inversion's depth to conductor, leave one inverted line out
    target = (
        con.execute(
            "SELECT sounding_id, conductor_top_m FROM model_conductor_depth "
            "WHERE run_id = ? AND conductor_top_m IS NOT NULL",
            [INVERTED_RUN],
        )
        .df()
        .set_index("sounding_id")["conductor_top_m"]
    )
    lines = survey.set_index("sounding_id")["line"]
    sets = feature_sets(survey, feats, feats_no_h)
    cv_rows, cv_pred = {}, {}
    for name, X in sets.items():
        pred = leave_one_line_out(X, target, lines)
        err = pred - target
        cv_pred[name] = pred
        cv_rows[name] = {
            "MAE_m": err.abs().mean(),
            "within_5m": (err.abs() <= 5).mean(),
            "R2": 1 - (err**2).sum() / ((target - target.mean()) ** 2).sum(),
        }
    cv = pd.DataFrame(cv_rows).T
    print(
        "\n3. depth to conductor, leave one of the 6 inverted lines out"
        f" ({len(target):,} soundings):"
    )
    print(cv.round(2).to_string())

    # 4. out of sample: lines without an inversion, against boreholes
    inverted_ids = con.execute(
        "SELECT DISTINCT sounding_id FROM model_resistivity WHERE run_id = ?", [INVERTED_RUN]
    ).df()["sounding_id"]
    uninverted = survey.loc[~survey["sounding_id"].isin(inverted_ids), "sounding_id"]
    predictions = {}
    for name in ("raw channels + height", "Kalman features"):
        X = sets[name]
        qs = fit_quantiles(X, target)
        predictions[name] = pd.DataFrame(
            {f"q{int(q * 100)}": m.predict(X.loc[uninverted]) for q, m in qs.items()},
            index=uninverted,
        )
    holes = con.sql(
        """
        SELECT a.hole_id, a.bedrock_depth_m, a.bedrock_unit, bs.sounding_id, bs.distance_m
        FROM aquifer_base AS a JOIN borehole_sounding AS bs USING (hole_id)
        WHERE a.reached_bedrock
        QUALIFY row_number() OVER (PARTITION BY a.hole_id ORDER BY bs.distance_m) = 1
        """
    ).df()
    holes = holes[(holes["distance_m"] <= 100) & holes["sounding_id"].isin(uninverted)].copy()
    holes["line"] = holes["sounding_id"].map(lines)
    kal = predictions["Kalman features"].loc[holes["sounding_id"]]
    for name, p in predictions.items():
        holes[f"{name}: q50"] = p.loc[holes["sounding_id"], "q50"].to_numpy().round(1)
    holes["Kalman q10"] = kal["q10"].to_numpy().round(0)
    holes["Kalman q90"] = kal["q90"].to_numpy().round(0)
    holes = holes.sort_values(["line", "bedrock_depth_m"])
    print(f"\n4. boreholes within 100 m of the uninverted lines ({len(holes)}):")
    print(holes.drop(columns=["sounding_id"]).round(1).to_string(index=False))
    summary = {
        name: {
            "MAE_m": (holes[f"{name}: q50"] - holes["bedrock_depth_m"]).abs().mean(),
            "median_abs_m": (holes[f"{name}: q50"] - holes["bedrock_depth_m"]).abs().median(),
        }
        for name in predictions
    }
    print(pd.DataFrame(summary).T.round(1).to_string())
    inside = holes["bedrock_depth_m"].between(holes["Kalman q10"], holes["Kalman q90"])
    print(f"bedrock depth inside the Kalman 80% interval at {inside.mean():.0%} of these holes")

    # maps and a section need predictions everywhere: CV on inverted lines, fitted model elsewhere
    full = pd.concat([cv_pred["Kalman features"], predictions["Kalman features"]["q50"]])
    fig_dir.mkdir(parents=True, exist_ok=True)
    everything = survey[["sounding_id", "line", "x", "y", "bird_height_m", "powerline_monitor"]]
    bedrock = con.sql(
        "SELECT hole_id, x, y, bedrock_depth_m FROM aquifer_base WHERE reached_bedrock"
    ).df()
    figures.kalman_depth_map(everything, full, bedrock, fig_dir / "kalman_conductor_depth.png")
    line = (
        int(holes["line"].value_counts().idxmax()) if len(holes) else int(survey["line"].iloc[-1])
    )
    figures.kalman_line(
        everything, feats, predictions["Kalman features"], holes, line, fig_dir / "kalman_line.png"
    )
    print(f"\nline section drawn for line {line}")
    return {"cv": cv, "holes": holes, "model": model}
