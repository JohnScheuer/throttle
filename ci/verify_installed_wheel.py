"""Confirm a clean-install test imported Throttle from site-packages."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import throttle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()

    package_path = Path(throttle.__file__).resolve()
    workspace = args.workspace.resolve()
    if package_path.is_relative_to(workspace):
        raise SystemExit(f"clean install imported repository source: {package_path}")
    if "site-packages" not in package_path.parts:
        raise SystemExit(f"clean install did not import from site-packages: {package_path}")

    installed_version = importlib.metadata.version("throttle-bench")
    if installed_version != throttle.__version__:
        raise SystemExit(
            "wheel metadata/package version mismatch: "
            f"{installed_version!r} != {throttle.__version__!r}"
        )

    print(f"clean wheel import: throttle {throttle.__version__} from site-packages")


if __name__ == "__main__":
    main()
