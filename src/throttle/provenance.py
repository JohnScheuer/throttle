"""Platform-aware runtime provenance for persisted benchmark manifests."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from .models import RunConfig

CURRENT_MANIFEST_VERSION = "1.1"
SUPPORTED_MANIFEST_VERSIONS = frozenset({"1.0", CURRENT_MANIFEST_VERSION})
ACCELERATOR_BACKENDS = ("cuda", "metal", "rocm", "cpu")
IMMUTABLE_ARTIFACT = re.compile(r"^(?:[^\s]+@)?sha256:[0-9a-f]{64}$")

LEGACY_RUNTIME_CONTROLLED_PATHS: tuple[tuple[str, ...], ...] = (
    ("runtime", "image_digest"),
    ("runtime", "gpu"),
    ("runtime", "gpu_fingerprint_sha256"),
    ("runtime", "gpu_fingerprint_supplied"),
    ("runtime", "cuda_version"),
    ("runtime", "driver_version"),
)

PLATFORM_RUNTIME_CONTROLLED_PATHS: tuple[tuple[str, ...], ...] = (
    *LEGACY_RUNTIME_CONTROLLED_PATHS,
    ("runtime", "accelerator_backend"),
    ("runtime", "accelerator"),
    ("runtime", "accelerator_fingerprint_sha256"),
    ("runtime", "accelerator_fingerprint_supplied"),
    ("runtime", "accelerator_runtime_version"),
    ("runtime", "host_os_version"),
    ("runtime", "software_environment_digest"),
)


def _public_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value.lower() != "unknown"
    )


def is_immutable_artifact_digest(value: object) -> bool:
    return isinstance(value, str) and bool(IMMUTABLE_ARTIFACT.fullmatch(value))


def build_runtime_manifest(config: RunConfig) -> dict[str, Any]:
    """Build manifest 1.1 runtime metadata while retaining legacy CUDA keys."""

    fingerprint_sha256 = hashlib.sha256(
        config.gpu_fingerprint.encode("utf-8")
    ).hexdigest()
    accelerator_runtime_version = config.accelerator_runtime_version
    software_environment_digest = config.software_environment_digest
    if config.accelerator_backend == "cuda":
        if accelerator_runtime_version == "unknown":
            accelerator_runtime_version = config.cuda_version
        if software_environment_digest == "unknown":
            software_environment_digest = config.image_digest
    return {
        # Manifest 1.0 compatibility fields. They stay explicit so a new reader can
        # still distinguish a real CUDA pin from a platform where CUDA is absent.
        "image_digest": config.image_digest,
        "gpu": config.gpu,
        "gpu_fingerprint_sha256": fingerprint_sha256,
        "gpu_fingerprint_supplied": config.gpu_fingerprint != "unknown",
        "cuda_version": config.cuda_version,
        "driver_version": config.driver_version,
        # Manifest 1.1 platform-neutral fields.
        "accelerator_backend": config.accelerator_backend,
        "accelerator": config.gpu,
        "accelerator_fingerprint_sha256": fingerprint_sha256,
        "accelerator_fingerprint_supplied": config.gpu_fingerprint != "unknown",
        "accelerator_runtime_version": accelerator_runtime_version,
        "host_os_version": config.host_os_version,
        "software_environment_digest": software_environment_digest,
    }


def runtime_provenance_reasons(
    runtime: object, manifest_version: object
) -> list[str]:
    """Return fixed, sanitized decision-gate reasons for runtime provenance."""

    if not isinstance(runtime, Mapping):
        return ["complete_runtime_provenance_required"]
    if manifest_version == "1.0":
        reasons: list[str] = []
        if any(
            not _public_text(runtime.get(field))
            for field in ("image_digest", "gpu", "cuda_version", "driver_version")
        ):
            reasons.append("complete_runtime_provenance_required")
        fingerprint = runtime.get("gpu_fingerprint_sha256")
        if not isinstance(fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", fingerprint
        ):
            reasons.append("invalid_gpu_fingerprint")
        elif runtime.get("gpu_fingerprint_supplied") is not True:
            reasons.append("gpu_fingerprint_not_supplied")
        if not is_immutable_artifact_digest(runtime.get("image_digest")):
            reasons.append("immutable_image_digest_required")
        return list(dict.fromkeys(reasons))
    if manifest_version != CURRENT_MANIFEST_VERSION:
        return ["unsupported_runtime_manifest_version"]

    backend = runtime.get("accelerator_backend")
    if backend not in ACCELERATOR_BACKENDS:
        return ["complete_runtime_provenance_required"]
    reasons = []
    required = ("accelerator", "accelerator_runtime_version")
    if backend != "cuda":
        required = (*required, "host_os_version")
    if any(not _public_text(runtime.get(field)) for field in required):
        reasons.append("complete_runtime_provenance_required")

    fingerprint = runtime.get("accelerator_fingerprint_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ):
        reasons.append("invalid_accelerator_fingerprint")
    elif runtime.get("accelerator_fingerprint_supplied") is not True:
        reasons.append("accelerator_fingerprint_not_supplied")

    if not is_immutable_artifact_digest(
        runtime.get("software_environment_digest")
    ):
        reasons.append("immutable_software_environment_digest_required")
    if backend == "cuda":
        if any(
            not _public_text(runtime.get(field))
            for field in ("image_digest", "cuda_version", "driver_version")
        ):
            reasons.append("complete_runtime_provenance_required")
        if not is_immutable_artifact_digest(runtime.get("image_digest")):
            reasons.append("immutable_image_digest_required")
    return list(dict.fromkeys(reasons))


def runtime_preflight_reason(runtime: object, manifest_version: object) -> str | None:
    reasons = runtime_provenance_reasons(runtime, manifest_version)
    if not reasons:
        return None
    reason = reasons[0]
    if reason == "complete_runtime_provenance_required":
        return "unverified_manifest_metadata"
    if reason in {"invalid_gpu_fingerprint", "invalid_accelerator_fingerprint"}:
        return "invalid_manifest_digest"
    return reason
