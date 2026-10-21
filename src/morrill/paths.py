"""Where things live on disk. Everything under data/ is gitignored."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# MORRILL_DATA_DIR lets you keep the (large) data outside the repo.
DATA_DIR = Path(os.environ.get("MORRILL_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "morrill.duckdb"

MANIFEST_PATH = REPO_ROOT / "manifests" / "raw_files.csv"
