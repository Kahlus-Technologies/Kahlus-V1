"""Frozen manifest of the external pretrained EEG checkpoints, with fail-closed hashing.

``docs/research/eeg_fm_benchmark_plan_2026-08-18.md`` section 9 requires a
``checkpoints.json`` artifact recording "source URL, SHA-256, licence, and mirror
location for every external checkpoint used", and section 8 makes "EEGPT figshare
checkpoint mirrored ... with a recorded SHA-256 **before** the run" a hard blocker. This
module is that record, frozen in code so a checkpoint swap is a diff rather than a
silent change of experiment.

Hard rules:

- **No network access.** Nothing here downloads, resolves, or contacts a host. The
  ``source_url`` fields are provenance strings for the artifact, not fetch targets, and
  ``huggingface_hub`` must never be imported (it is quarantined repo-wide by
  ``tests/artifacts/test_docs_contracts.py``). Staging weights is a deliberate,
  out-of-band, human-run step.
- **Fail closed.** :func:`resolve_checkpoint` returns a path only after the file exists
  *and* its SHA-256 matches the frozen digest. A missing file, a truncated file, or a
  different revision of "the same" checkpoint all raise.
- **Slate discipline.** Only checkpoints admissible under the benchmark slate appear
  here. The BIOT ``six-datasets`` checkpoint is deliberately absent: it is in
  ``pretraining_contamination.SLATE_EXCLUDED_MODEL_IDS`` and leaving it out of the
  manifest removes the possibility of loading it by mistake, even though the file ships
  in the same upstream directory as the admissible ``PREST-16`` one.

``model_id`` values are the canonical repo ids used by
``neurotwin.benchmarks.pretraining_contamination`` (``labram_base``, ``biot_prest16``),
so a manifest row joins directly against a contamination row. Lookups also accept the
``upstream_id`` alias (``labram``, ``biot``) that names the vendor directory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


_HASH_CHUNK_BYTES = 1 << 20


class WeightsManifestError(RuntimeError):
    """Base class for checkpoint staging and verification failures."""


class CheckpointNotStagedError(WeightsManifestError):
    """Raised when a manifest checkpoint is not present in the weights directory."""


class CheckpointChecksumError(WeightsManifestError):
    """Raised when a staged checkpoint's SHA-256 does not match the frozen digest."""


@dataclass(frozen=True)
class CheckpointSpec:
    """One staged external checkpoint.

    Attributes:
        model_id: canonical repo model id, joinable against ``pretraining_contamination``.
        upstream_id: ``external/upstreams.lock.json`` key / vendor directory name.
        filename: expected filename inside the staged weights directory.
        sha256: frozen digest of the exact bytes this benchmark is allowed to run.
        source_url: provenance only; never fetched by this module.
        license_note: weight licence, which for EEGPT differs from its code licence.
        upstream_relpath: path inside the vendored upstream checkout when the checkpoint
            ships in-repo, else ``None`` (weights hosted off-repo and mirrored by hand).
    """

    model_id: str
    upstream_id: str
    filename: str
    sha256: str
    source_url: str
    license_note: str
    upstream_relpath: str | None = None

    def as_row(self) -> dict[str, str]:
        """Artifact row for ``artifacts/eeg_fm_benchmark/checkpoints.json``."""

        return {
            "model_id": self.model_id,
            "upstream_id": self.upstream_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "source_url": self.source_url,
            "license_note": self.license_note,
            "upstream_relpath": self.upstream_relpath or "",
        }


WEIGHTS_MANIFEST: tuple[CheckpointSpec, ...] = (
    CheckpointSpec(
        model_id="eegpt",
        upstream_id="eegpt",
        filename="eegpt_mcae_58chs_4s_large4E.ckpt",
        sha256="9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0",
        source_url="https://figshare.com/s/e37df4f8a907a866df4b",
        license_note=(
            "Weights CC BY 4.0 (figshare item); upstream code Apache-2.0. Attribution is "
            "required when redistributing or deriving from the weights. Hosted only as a "
            "figshare share link with no redundancy, so this benchmark runs against a "
            "locally mirrored copy pinned by the digest above (plan section 8)."
        ),
    ),
    CheckpointSpec(
        model_id="cbramod",
        upstream_id="cbramod",
        filename="cbramod_pretrained_weights.pth",
        sha256="0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
        source_url="https://huggingface.co/weighting666/CBraMod",
        license_note=(
            "Weights Apache-2.0 per the model card; upstream code MIT. Hosted off-repo on "
            "Hugging Face and staged by hand - huggingface_hub is quarantined in this "
            "repo and must not be used to fetch it."
        ),
    ),
    CheckpointSpec(
        model_id="labram_base",
        upstream_id="labram",
        filename="labram-base.pth",
        sha256="7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c",
        source_url="https://github.com/935963004/LaBraM/tree/c431221e6cfd23dbfa9950e0180682fb322b0548/checkpoints",
        license_note=(
            "MIT (upstream repo licence covers the in-repo checkpoint). Only the Base "
            "checkpoint is public; Large and Huge are withheld by the authors."
        ),
        upstream_relpath="checkpoints/labram-base.pth",
    ),
    CheckpointSpec(
        model_id="biot_prest16",
        upstream_id="biot",
        filename="biot-EEG-PREST-16-channels.ckpt",
        sha256="40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70",
        source_url="https://github.com/ycq091044/BIOT/tree/d138e32634e52ae9fa6ec98ac9c4087b14ca869a/pretrained-models",
        license_note=(
            "MIT (upstream repo licence covers the in-repo checkpoints). Staged under a "
            "'biot-' prefixed filename so the admissible PREST-16 checkpoint cannot be "
            "confused with the slate-excluded six-datasets checkpoint that ships beside it."
        ),
        upstream_relpath="pretrained-models/EEG-PREST-16-channels.ckpt",
    ),
)

_SPEC_INDEX: dict[str, CheckpointSpec] = {}
for _spec in WEIGHTS_MANIFEST:
    _SPEC_INDEX[_spec.model_id] = _spec
    _SPEC_INDEX.setdefault(_spec.upstream_id, _spec)


def manifest_model_ids() -> tuple[str, ...]:
    """Canonical ``model_id`` values in manifest order."""

    return tuple(spec.model_id for spec in WEIGHTS_MANIFEST)


def manifest_rows() -> list[dict[str, str]]:
    """The whole manifest as machine-readable rows for the run artifact."""

    return [spec.as_row() for spec in WEIGHTS_MANIFEST]


def checkpoint_spec(model_id: str) -> CheckpointSpec:
    """Return the frozen spec for ``model_id`` (canonical id or ``upstream_id`` alias)."""

    spec = _SPEC_INDEX.get(str(model_id))
    if spec is None:
        known = ", ".join(manifest_model_ids())
        raise WeightsManifestError(
            f"no staged checkpoint is registered for model_id {model_id!r}. "
            f"Registered model ids: {known}. Checkpoints outside this manifest are not "
            "admissible in the benchmark; the BIOT six-datasets checkpoint is excluded "
            "on purpose (pretraining_contamination.SLATE_EXCLUDED_MODEL_IDS)."
        )
    return spec


def sha256_file(path: Path | str) -> str:
    """Streaming SHA-256 of a file, so a 92 MB checkpoint is not read into memory whole."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path | str, spec: CheckpointSpec) -> Path:
    """Verify ``path`` against ``spec`` and return it, or raise.

    Fails closed on both "absent" and "present but different bytes"; there is no
    "warn and continue" branch, because a silently mismatched checkpoint changes the
    experiment without changing the code.
    """

    candidate = Path(path)
    if not candidate.is_file():
        raise CheckpointNotStagedError(
            f"checkpoint for {spec.model_id!r} is not staged at {candidate}. "
            f"Stage {spec.filename!r} there by hand from {spec.source_url} "
            f"(licence: {spec.license_note.splitlines()[0]}). "
            "Nothing in this repo downloads weights."
        )
    observed = sha256_file(candidate)
    if observed != spec.sha256:
        raise CheckpointChecksumError(
            f"checkpoint for {spec.model_id!r} at {candidate} has sha256 {observed}, "
            f"expected {spec.sha256}. Refusing to load: the staged file is truncated, "
            "corrupted, or a different revision than the one this benchmark is pinned to."
        )
    return candidate


def resolve_checkpoint(model_id: str, weights_dir: Path | str) -> Path:
    """Return the verified path to ``model_id``'s checkpoint inside ``weights_dir``.

    Offline and fail-closed: the file must already exist and must hash to the frozen
    digest. Callers get a path they can hand to ``torch.load`` knowing it is the exact
    revision recorded in the run artifact.
    """

    spec = checkpoint_spec(model_id)
    root = Path(weights_dir)
    if not root.is_dir():
        raise CheckpointNotStagedError(
            f"weights directory {root} does not exist, so {spec.filename!r} for "
            f"{spec.model_id!r} cannot be resolved. Create it and stage the checkpoint "
            "there by hand; nothing in this repo downloads weights."
        )
    return verify_checkpoint(root / spec.filename, spec)
