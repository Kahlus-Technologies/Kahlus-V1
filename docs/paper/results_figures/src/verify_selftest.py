"""Self-test: prove each check in verify_figures.py can actually fail.

A verifier that passes everything is indistinguishable from no verifier. Each
case here plants exactly one fault of the kind its check exists to catch and
asserts that check reports FAIL. Nothing is mutated on disk: manifests are
copied in memory and the artifact root is redirected at a temp tree.

Run:  python3 verify_selftest.py
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import verify_figures as V


def manifest(name: str) -> dict:
    return json.loads((V.FIGDIR / f"{name}.json").read_text())


def failed(fn, *args) -> bool:
    """Run one check in isolation and report whether it failed."""
    res = V.Result()
    fn(*args, res)
    return bool(res.failed)


CASES: list[tuple[str, str]] = []


def case(label: str):
    def deco(fn):
        CASES.append((label, fn))
        return fn
    return deco


@case("NUMBERS catches a hand-typed series value")
def t_numbers_typed() -> bool:
    m = manifest("fig_crossover")
    m["series"]["kahlus_gru"][0] = 0.1234567
    return failed(V.check_numbers, "fig_crossover", m)


@case("NUMBERS catches a derived value that disagrees with the artifact")
def t_numbers_derived() -> bool:
    m = manifest("fig_crossover")
    # Plausible-looking but wrong: 10 ms becomes 12 ms.
    m["horizons_ms"][0] = 12.0
    return failed(V.check_numbers, "fig_crossover", m)


@case("NUMBERS catches a stale count after the artifact changes")
def t_numbers_count() -> bool:
    m = manifest("fig_overlap_law")
    m["n_synthetic_cells"] = 103
    return failed(V.check_numbers, "fig_overlap_law", m)


@case("LEGIBILITY catches type that would print under the floor")
def t_legibility() -> bool:
    m = manifest("fig_crossover")
    # A figure authored at 3.25 in but placed at 1.6 in halves every glyph.
    m["figure_width_in"] = 6.5
    return failed(V.check_legibility, "fig_crossover", m)


@case("WIDTH catches a tight-bbox canvas that grew past the column")
def t_width() -> bool:
    m = manifest("fig_crossover")
    m["figure_width_in"] = 4.08          # the exact bbox="tight" regression
    return failed(V.check_width, "fig_crossover", m)


@case("DPI catches a raster below 300 dpi at placement")
def t_dpi() -> bool:
    from PIL import Image
    real = V.FIGDIR
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for ext in ("json", "pdf"):
            shutil.copy(real / f"fig_crossover.{ext}", tmp)
        im = Image.open(real / "fig_crossover.png")
        w = int(3.25 * 150)              # 150 dpi at placement
        im.resize((w, int(im.size[1] * w / im.size[0]))).save(
            tmp / "fig_crossover.png")
        V.FIGDIR = tmp
        try:
            return failed(V.check_dpi, "fig_crossover", manifest("fig_crossover"))
        finally:
            V.FIGDIR = real


@case("RETRACTION catches a retracted number outside a retraction figure")
def t_retraction() -> bool:
    m = manifest("fig_crossover")
    m["headline_correlation"] = "r = 0.972"
    return failed(V.check_retraction, "fig_crossover", m)


@case("RETRACTION still allows the figure whose subject is the retraction")
def t_retraction_allowed() -> bool:
    m = manifest("fig_overlap_law")
    m["note"] = "explains the retracted 3.116"
    return not failed(V.check_retraction, "fig_overlap_law", m)


@case("PROVENANCE catches an artifact that changed since the last run")
def t_provenance() -> bool:
    m = manifest("fig_crossover")
    prov = {"fig_crossover": {m["artifact"]: "0" * 64}}
    res = V.Result()
    V.check_provenance("fig_crossover", m, res, prov)
    return bool(res.failed)


@case("PROVENANCE catches a missing artifact")
def t_provenance_missing() -> bool:
    m = manifest("fig_crossover")
    m["artifact"] = "no_such_run/results.json"
    res = V.Result()
    V.check_provenance("fig_crossover", m, res, {})
    return bool(res.failed)


@case("NO-REGRESS catches a modified committed figure")
def t_frozen() -> bool:
    real_root, real_frozen = V.ROOT, V.FROZEN
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # A temp dir is not a git repo, so `git status` cannot vouch for the
        # files; the check must refuse to pass rather than assume clean.
        dst = tmp / "docs/paper/vertical_figures/figures_final"
        dst.mkdir(parents=True)
        for p in real_frozen:
            shutil.copy(p, dst)
        V.ROOT = tmp
        V.FROZEN = sorted(dst.glob("*.png"))
        try:
            res = V.Result()
            V.check_frozen(res, {})
            return bool(res.failed)
        finally:
            V.ROOT, V.FROZEN = real_root, real_frozen


@case("DATASET catches a BNCI caption drawn from a Sleep-EDF artifact")
def t_dataset() -> bool:
    m = manifest("fig_crossover")
    m["dataset"] = "BNCI2014_001"        # artifact path still says sleep-edf
    return failed(V.check_dataset, "fig_crossover", m)


@case("DATASET catches a figure that claims no dataset and states no law")
def t_dataset_silent() -> bool:
    m = manifest("fig_crossover")
    del m["dataset"]
    return failed(V.check_dataset, "fig_crossover", m)


def main() -> int:
    ok = 0
    for label, fn in CASES:
        try:
            caught = fn()
        except Exception as exc:                     # noqa: BLE001
            print(f"  ERROR {label}: {exc!r}")
            continue
        print(f"  {'PASS' if caught else 'FAIL'}  {label}")
        ok += bool(caught)
    print(f"\n{ok}/{len(CASES)} self-tests pass")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
