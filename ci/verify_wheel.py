"""Compare every packaged Throttle source/data file with the source tree."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath


def _source_files(source_root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root.parent)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files[relative.as_posix()] = path.read_bytes()
    return files


def _wheel_files(archive: zipfile.ZipFile) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise SystemExit(f"unsafe path in wheel: {info.filename!r}")
        if info.is_dir() or not path.parts or path.parts[0] != "throttle":
            continue
        if info.filename in files:
            raise SystemExit(f"duplicate package entry in wheel: {info.filename}")
        files[info.filename] = archive.read(info)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "throttle",
    )
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    source_root = args.source_root.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SystemExit(f"not a wheel file: {wheel}")
    if not source_root.is_dir():
        raise SystemExit(f"source package not found: {source_root}")

    expected = _source_files(source_root)
    if not expected:
        raise SystemExit(f"source package has no files: {source_root}")
    with zipfile.ZipFile(wheel) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise SystemExit(f"wheel CRC check failed for: {bad_member}")
        actual = _wheel_files(archive)

    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(
        path
        for path in expected.keys() & actual.keys()
        if expected[path] != actual[path]
    )
    if missing or unexpected or changed:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if changed:
            details.append(f"byte_mismatch={changed}")
        raise SystemExit("wheel/source mismatch: " + "; ".join(details))

    print(f"wheel/source parity: {len(expected)} package files match byte-for-byte")


if __name__ == "__main__":
    main()
