"""Lazy, fail-loud access to the vendored upstream EEG model repositories.

``external/vendor/`` is gitignored: it is populated on demand by
``scripts/vendor_upstreams.py`` from the pinned commits in
``external/upstreams.lock.json``. It is therefore **absent on a fresh checkout and in
CI**, which means:

- nothing in this repo may import vendored code at module import time;
- every consumer must go through :func:`require_vendored` inside the function that
  actually needs the upstream module, so a missing vendor tree produces one clear,
  actionable error instead of an ``ImportError`` from three frames deep;
- tests that touch real upstream code must *skip*, not fail, when the tree is absent
  (see :func:`is_vendored`).

Three of the four EEG entrants (``labram``, ``biot``, and the ``mixed``-status extras) are
marked ``reuse_status="mixed"`` in the lock file, so vendoring them requires the explicit
``--include-restricted`` flag. The error message below names that flag, because omitting
it is the most common way to end up with a partially populated vendor tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


#: Upstream ids for the four external pretrained EEG entrants, as keyed in
#: ``external/upstreams.lock.json`` and therefore as named on disk under
#: ``external/vendor/``. These are *upstream* ids, not the ``model_id`` values used by
#: ``pretraining_contamination`` (``labram`` here is ``labram_base`` there).
VENDORED_UPSTREAM_IDS: tuple[str, ...] = ("eegpt", "cbramod", "labram", "biot")

#: Environment override for the vendor tree location, so a cluster job can point at a
#: read-only shared checkout without moving the repo.
VENDOR_DIR_ENV_VAR = "NEUROTWIN_VENDOR_DIR"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_VENDOR_DIR = _REPO_ROOT / "external" / "vendor"


class VendoredUpstreamMissing(RuntimeError):
    """Raised when a vendored upstream repository is not present on disk."""


def vendor_dir(explicit: Path | str | None = None) -> Path:
    """Return the root of the vendor tree.

    Resolution order: explicit argument, then ``$NEUROTWIN_VENDOR_DIR``, then
    ``<repo>/external/vendor``.
    """

    if explicit is not None:
        return Path(explicit)
    override = os.environ.get(VENDOR_DIR_ENV_VAR)
    if override:
        return Path(override)
    return _DEFAULT_VENDOR_DIR


def vendored_root(upstream_id: str, *, vendor_dir_path: Path | str | None = None) -> Path | None:
    """Return the vendored checkout for ``upstream_id``, or ``None`` if it is unusable.

    "Unusable" covers both "absent" and "present but empty", the latter being what a
    half-finished or interrupted clone leaves behind.
    """

    candidate = vendor_dir(vendor_dir_path) / str(upstream_id)
    if not candidate.is_dir():
        return None
    if not any(candidate.iterdir()):
        return None
    return candidate


def is_vendored(upstream_id: str, *, vendor_dir_path: Path | str | None = None) -> bool:
    """True when ``upstream_id`` has a usable vendored checkout.

    Intended for ``unittest.skipUnless`` guards: a missing vendor tree is an environment
    fact, not a test failure.
    """

    return vendored_root(upstream_id, vendor_dir_path=vendor_dir_path) is not None


def vendor_command(upstream_id: str) -> str:
    """The exact command that populates ``upstream_id``'s vendored checkout."""

    return f"python3 scripts/vendor_upstreams.py --ids {upstream_id} --include-restricted"


def require_vendored(upstream_id: str, *, vendor_dir_path: Path | str | None = None) -> Path:
    """Return the vendored checkout for ``upstream_id`` or raise a directive error."""

    root = vendored_root(upstream_id, vendor_dir_path=vendor_dir_path)
    if root is not None:
        return root
    expected = vendor_dir(vendor_dir_path) / str(upstream_id)
    known = ", ".join(VENDORED_UPSTREAM_IDS)
    raise VendoredUpstreamMissing(
        f"vendored upstream {upstream_id!r} is not available at {expected}. "
        f"external/vendor/ is gitignored and must be populated locally: run "
        f"`{vendor_command(upstream_id)}`. "
        f"Known external EEG model upstream ids: {known}. "
        "Do not import vendored code at module import time."
    )


def ensure_on_sys_path(upstream_id: str, *, vendor_dir_path: Path | str | None = None) -> Path:
    """Require ``upstream_id`` and prepend its checkout to ``sys.path``.

    Idempotent: repeated calls do not stack duplicate entries. Upstream EEG repos are
    scripts-with-flat-imports, not installable packages, so their root has to be on
    ``sys.path`` for ``import modeling_finetune`` (LaBraM) or ``from model import BIOT``
    to resolve. Call this as late as possible - it mutates global interpreter state, and
    several of these repos shadow common module names (``model``, ``utils``, ``datasets``).
    """

    root = require_vendored(upstream_id, vendor_dir_path=vendor_dir_path)
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root
