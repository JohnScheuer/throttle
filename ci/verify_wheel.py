"""Compare every packaged Throttle source/data file with the source tree."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import tomllib
import zipfile
from email import policy
from email.parser import BytesParser
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


def _single_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one wheel member ending in {suffix!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _verify_metadata(archive: zipfile.ZipFile, project_root: Path) -> None:
    with (project_root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    expected_name = str(project["name"])
    expected_version = str(project["version"])

    metadata_name = _single_member(archive, ".dist-info/METADATA")
    metadata = BytesParser(policy=policy.default).parsebytes(
        archive.read(metadata_name)
    )
    if metadata["Name"] != expected_name or metadata["Version"] != expected_version:
        raise SystemExit("wheel metadata name/version does not match pyproject.toml")

    entry_points_name = _single_member(archive, ".dist-info/entry_points.txt")
    entry_points = configparser.ConfigParser()
    entry_points.read_string(archive.read(entry_points_name).decode("utf-8"))
    if entry_points.get("console_scripts", "throttle", fallback=None) != (
        "throttle.cli:main"
    ):
        raise SystemExit("wheel console entry point is missing or incorrect")

    wheel_name = _single_member(archive, ".dist-info/WHEEL")
    wheel_metadata = BytesParser(policy=policy.default).parsebytes(
        archive.read(wheel_name)
    )
    if "py3-none-any" not in wheel_metadata.get_all("Tag", []):
        raise SystemExit("wheel is not tagged py3-none-any")

    license_members = [
        name
        for name in archive.namelist()
        if ".dist-info/licenses/" in name and name.endswith("/LICENSE")
    ]
    if len(license_members) != 1:
        raise SystemExit("wheel must contain exactly one packaged LICENSE")
    if archive.read(license_members[0]) != (project_root / "LICENSE").read_bytes():
        raise SystemExit("packaged LICENSE does not match repository LICENSE")


def _verify_record(archive: zipfile.ZipFile) -> None:
    record_name = _single_member(archive, ".dist-info/RECORD")
    rows = list(
        csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
    )
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or row[0] in records:
            raise SystemExit("wheel RECORD is malformed or contains duplicates")
        records[row[0]] = (row[1], row[2])

    members = {
        info.filename: archive.read(info)
        for info in archive.infolist()
        if not info.is_dir()
    }
    if set(records) != set(members):
        raise SystemExit("wheel RECORD members do not match archive members")
    for name, data in members.items():
        digest, size = records[name]
        if name == record_name:
            if digest or size:
                raise SystemExit("wheel RECORD must leave its own hash and size empty")
            continue
        if not digest.startswith("sha256="):
            raise SystemExit("wheel RECORD contains a non-SHA-256 digest")
        encoded_digest = digest.removeprefix("sha256=")
        padding = "=" * (-len(encoded_digest) % 4)
        try:
            recorded_digest = base64.urlsafe_b64decode(encoded_digest + padding)
        except ValueError as exc:
            raise SystemExit("wheel RECORD contains an invalid digest") from exc
        if recorded_digest != hashlib.sha256(data).digest():
            raise SystemExit(f"wheel RECORD digest mismatch: {name}")
        if size != str(len(data)):
            raise SystemExit(f"wheel RECORD size mismatch: {name}")


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
        _verify_metadata(archive, source_root.parents[1])
        _verify_record(archive)
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
