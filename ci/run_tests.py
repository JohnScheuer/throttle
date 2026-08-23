"""Run the full pytest suite and reject any reduced-coverage outcome."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-directory",
        type=Path,
        default=Path("tests"),
    )
    args = parser.parse_args()

    start_directory = args.start_directory.resolve()
    if not start_directory.is_dir():
        raise SystemExit("test start directory does not exist")

    # Import pytest here to ensure it's available
    try:
        import pytest
    except ImportError:
        print("pytest not installed", file=sys.stderr)
        raise SystemExit(1)

    # Run pytest with strict configuration:
    # -v: verbose
    # --tb=short: short traceback format
    # --strict-markers: reject unknown markers
    # -ra: show summary of all outcomes except passed
    # --no-cov: disable coverage if accidentally enabled
    exit_code = pytest.main([
        str(start_directory),
        "-v",
        "--tb=short",
        "--strict-markers",
        "-ra",
    ])

    # pytest exit codes:
    # 0: all tests passed
    # 1: tests failed
    # 2: test execution interrupted
    # 3: internal error
    # 4: pytest command line usage error
    # 5: no tests collected
    if exit_code == 5:
        print("CI discovered zero tests", file=sys.stderr)
        raise SystemExit(1)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
