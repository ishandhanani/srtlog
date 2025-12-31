#!/usr/bin/env python3
"""
srtlog - Log analysis CLI for SGLang distributed inference benchmarks
"""

import argparse
import sys

from .parse import add_parse_subparser


def main():
    """Main entry point for srtlog CLI."""
    parser = argparse.ArgumentParser(
        prog="srtlog",
        description="Log analysis toolkit for SGLang distributed inference benchmarks",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    add_parse_subparser(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
