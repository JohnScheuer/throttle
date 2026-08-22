"""Verify throttle is installed from the expected directory.

This guard prevents stale editable installs from interfering with pre-fix
verification or fresh-clone testing. When testing against a specific commit,
throttle.__file__ must resolve inside the expected checkout directory.

Usage:
    # For testing current working directory install
    python ci/verify_install_location.py

    # For testing specific directory install
    python ci/verify_install_location.py --expected-dir /path/to/checkout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify throttle.__file__ resolves inside expected directory"
    )
    parser.add_argument(
        "--expected-dir",
        type=Path,
        default=None,
        help="Expected parent directory (defaults to current working directory)",
    )
    args = parser.parse_args()

    # Import throttle to get its __file__ location
    try:
        import throttle
    except ImportError:
        print(
            "ERROR: throttle package not installed",
            file=sys.stderr,
        )
        raise SystemExit(1)

    throttle_file = Path(throttle.__file__).resolve()

    # Determine expected directory
    if args.expected_dir is None:
        expected_dir = Path.cwd().resolve()
    else:
        expected_dir = args.expected_dir.resolve()

    # Check if throttle.__file__ is inside expected_dir
    try:
        throttle_file.relative_to(expected_dir)
    except ValueError:
        print(
            f"ERROR: Stale install detected\n"
            f"  Expected: throttle installed from {expected_dir}\n"
            f"  Actual:   throttle.__file__ = {throttle_file}\n"
            f"\n"
            f"This usually means:\n"
            f"  1. An editable install (.pth file) points to a different directory\n"
            f"  2. You're testing a specific commit but throttle is from another checkout\n"
            f"\n"
            f"To fix:\n"
            f"  - Create a fresh venv and install from the expected directory\n"
            f"  - Or uninstall throttle and reinstall with: pip install -e {expected_dir}\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"OK: throttle.__file__ = {throttle_file}")
    print(f"    Expected parent: {expected_dir}")
    print(f"    ✓ Install location verified")


if __name__ == "__main__":
    main()
