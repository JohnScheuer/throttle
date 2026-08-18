"""Run the full unittest suite and reject any reduced-coverage outcome."""

from __future__ import annotations

import argparse
import sys
import unittest
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

    suite = unittest.defaultTestLoader.discover(str(start_directory))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun == 0:
        print("CI discovered zero tests", file=sys.stderr)
        raise SystemExit(1)
    reduced_coverage = (
        len(result.skipped)
        + len(result.expectedFailures)
        + len(result.unexpectedSuccesses)
    )
    if reduced_coverage:
        print(
            "CI requires zero skipped, expected-failure, or unexpected-success "
            f"tests; observed {reduced_coverage}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
