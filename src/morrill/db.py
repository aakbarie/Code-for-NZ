"""Opening the database and creating its tables."""

from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

from morrill.paths import DB_PATH
from morrill.raw import sha256_of


def connect(path: Path | str = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=read_only)


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(resources.files("morrill").joinpath("schema.sql").read_text())


def load_lithology_classes(con: duckdb.DuckDBPyConnection) -> None:
    path = resources.files("morrill").joinpath("reference/lithology_classes.csv")
    with resources.as_file(path) as csv_path:
        classes = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    classes["consolidated"] = classes["consolidated"].map({"true": True, "false": False, "": None})
    classes["note"] = classes["note"].replace("", None)
    insert_df(con, "lithology_class", classes)


def register_source(
    con: duckdb.DuckDBPyConnection,
    source_id: str,
    raw_dir: Path,
    local_path: str,
    description: str,
) -> None:
    con.execute(
        "INSERT INTO source_file VALUES (?, ?, ?, ?)",
        [source_id, local_path, sha256_of(raw_dir / local_path), description],
    )


def insert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Append a DataFrame to a table, matching columns by name."""
    con.register("_incoming", df)
    try:
        cols = ", ".join(df.columns)
        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming")
    finally:
        con.unregister("_incoming")
    return len(df)
