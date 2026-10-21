"""Command-line entry point: `uv run morrill <command>`."""

import argparse
import sys

from morrill import build, raw
from morrill.paths import DATA_DIR, DB_PATH, REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="morrill")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-raw", help="check data/raw against manifests/raw_files.csv")
    sub.add_parser("build", help="build data/morrill.duckdb from data/raw")
    sub.add_parser(
        "communities",
        help="group soundings by resistivity profile (needs: uv sync --extra analysis)",
    )
    sub.add_parser(
        "lithology",
        help="rock-class probabilities from resistivity (needs: uv sync --extra analysis)",
    )
    sub.add_parser(
        "kalman",
        help="state-space features along flight lines (needs: uv sync --extra analysis)",
    )

    args = parser.parse_args(argv)
    if args.command == "verify-raw":
        return 0 if raw.verify() else 1
    if args.command == "build":
        build.build()
        return 0
    if args.command in ("communities", "lithology", "kalman"):
        try:
            from morrill.analysis import communities, kalman, lithology
        except ImportError as err:
            print(f"{err}. Install the analysis extra: uv sync --extra analysis")
            return 1
        analysis = {"communities": communities, "lithology": lithology, "kalman": kalman}[
            args.command
        ]
        analysis.run(DB_PATH, DATA_DIR / "derived", REPO_ROOT / "docs" / "figures")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
