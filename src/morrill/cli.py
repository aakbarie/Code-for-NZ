"""Command-line entry point: `uv run morrill <command>`."""

import argparse
import sys

from morrill import raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="morrill")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-raw", help="check data/raw against manifests/raw_files.csv")

    args = parser.parse_args(argv)
    if args.command == "verify-raw":
        return 0 if raw.verify() else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
