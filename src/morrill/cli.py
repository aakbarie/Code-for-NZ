"""Command-line entry point: `uv run morrill <command>`."""

import argparse
import sys

from morrill import build, raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="morrill")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-raw", help="check data/raw against manifests/raw_files.csv")
    sub.add_parser("build", help="build data/morrill.duckdb from data/raw")

    args = parser.parse_args(argv)
    if args.command == "verify-raw":
        return 0 if raw.verify() else 1
    if args.command == "build":
        build.build()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
