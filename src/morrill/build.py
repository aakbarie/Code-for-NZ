"""Build data/morrill.duckdb from the raw files, from scratch, in one go.

The database is never edited by hand: change a loader or a reference file and
rebuild. The build writes to a temporary file and only replaces the old
database once everything has loaded, so a failed build leaves nothing broken.
"""

from pathlib import Path

from morrill import db, link, qc
from morrill.ingest import airborne, boreholes, inversion, layers
from morrill.paths import DB_PATH, RAW_DIR

# Raw inputs used by the build, relative to data/raw (see manifests/raw_files.csv).
SOURCES = {
    "induction": "boreholes/CSD_ires.csv",
    "lithology": "boreholes/CSD_Lith.csv",
    "stratigraphy": "boreholes/CSD_strat.csv",
    "layers": "inversion/inv_model_pt1_bi.csv",
    "survey": "hem/NPblock2008_phaseCorr_fixed.csv",
    "inversion": "inversion/inv_model_pt1.csv",
}

# Known inputs not loaded yet, and why.
NOT_YET_LOADED = {
    "inversion/Morrill_block.res_XYZ_sparse_mod1.csv": "28-layer inversion: not retrieved",
    "inversion/i__con1.mod": "unidentified model file: not retrieved",
    "benchmark_2015/Morrill_block_test_*_voxler.csv": "2015 SOM results: not retrieved",
}


def build(
    db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR, sources: dict[str, str] | None = None
) -> list[tuple[str, bool, str]]:
    sources = sources or SOURCES
    missing = [p for p in sources.values() if not (raw_dir / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing raw files in {raw_dir}: {missing}. Run `morrill verify-raw` for details."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".building")
    tmp.unlink(missing_ok=True)

    con = db.connect(tmp)
    try:
        db.create_schema(con)
        db.load_reference(con)
        counts = layers.load(con, raw_dir, sources["layers"])
        counts |= boreholes.load(
            con, raw_dir, sources["induction"], sources["lithology"], sources["stratigraphy"]
        )
        counts |= airborne.load(con, raw_dir, sources["survey"])  # needs boreholes (datum)
        counts |= inversion.load(con, raw_dir, sources["inversion"])  # needs soundings
        counts |= link.link_boreholes(con)
        results = qc.run(con)
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    tmp.replace(db_path)

    print(f"Built {db_path}")
    for table, n in counts.items():
        print(f"  {table:<24}{n:>8,}")
    print("Checks:")
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'LOOK'}  {name}: {detail}")
    if NOT_YET_LOADED:
        print("Not loaded yet:")
        for path, why in NOT_YET_LOADED.items():
            print(f"  {path}: {why}")
    return results
