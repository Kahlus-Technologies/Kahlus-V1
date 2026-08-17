"""Verifier for the column-width results figures.

A figure is a claim, so it should be checkable the same way a result is. Each
check below exists because the corresponding failure has actually happened in
this repository at least once:

  1 NUMBERS      every plotted series is re-read from its artifact JSON now and
                 compared elementwise to the manifest the figure was built from.
                 A hand-typed number cannot survive this.
  2 LEGIBILITY   the PDF's own font records are read and the smallest type is
                 asserted >= MIN_PT at the placement width. Earlier work guessed
                 from glyph pixels and under-read.
  3 WIDTH        saved width must equal the placement width, or the type is
                 rescaled on the page and the authored point size is a lie.
  4 DPI          raster must be >= 300 dpi at placement.
  5 RETRACTION   the retracted 3.116 / 0.972 / 2.5x must not appear as a live
                 result. Allowed only where the figure is explicitly about the
                 retraction, which is declared per figure, not inferred.
  6 PROVENANCE   the artifact each figure names must exist and its sha256 is
                 recorded; drift since the last run is reported.
  7 NO-REGRESS   the five committed hybrid figures must be byte-identical.
  8 DATASET      each manifest names its dataset; a figure may not claim a
                 dataset it did not draw from.

Run:  python3 verify_figures.py          (exit 0 only if every check passes)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"
ROOT = HERE.parents[3]
ART = ROOT / "artifacts"
PROV = FIGDIR / "provenance.json"

COL_W = 3.25          # IEEE column width, inches
MIN_PT = 6.5          # smallest authored type; annotations
MIN_DPI = 300

RETRACTED = ("3.116", "0.972", "2.5x")

# Figures whose subject *is* the retraction. Anywhere else the retracted
# numbers are a live-result regression and fail the scan.
RETRACTION_CONTEXT = {"fig_overlap_law"}

# The committed hybrid figures that this work must not disturb.
FROZEN = sorted((ROOT / "docs/paper/vertical_figures/figures_final").glob("*.png"))


class Result:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, fig: str, check: str, ok: bool, msg: str = "") -> None:
        self.rows.append((fig, check, ok, msg))

    @property
    def failed(self) -> list[tuple[str, str, bool, str]]:
        return [r for r in self.rows if not r[2]]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten(obj, prefix: str = "") -> dict[str, float]:
    """Collect every leaf number in a manifest, keyed by its path."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def artifact_numbers(path: Path) -> set[float]:
    """Every number anywhere in an artifact, as a set for membership tests."""
    with path.open() as f:
        d = json.load(f)
    return {round(v, 10) for v in flatten(d).values()}


# --- check 1: numbers -------------------------------------------------------
# Structural keys the figure adds itself; they are not claims about data.
MANIFEST_META = {
    "figure_width_in", "figure_height_in", "min_authored_pt",
}


# Some plotted quantities are derived rather than copied: a horizon in ms is a
# horizon in samples over the sampling rate, skill is a ratio of two recorded
# MSEs. Loosening the membership test to let those through would defeat the
# check, because almost any number is within tolerance of some product. Instead
# each derived quantity is recomputed here, independently of the figure code,
# straight from the artifact. If the figure and the verifier disagree, that is
# the bug this check exists to find.
def derive_crossover(art: dict) -> dict[str, float]:
    fs = art["sampling_rate_hz"]
    hs = art["horizons"]
    ms = [h / fs * 1000.0 for h in hs]
    out = {f"horizons_ms[{i}]": v for i, v in enumerate(ms)}

    comp = art["comparisons"]
    gru = [comp[str(h)]["mean_mse"]["kahlus_gru"] for h in hs]
    ridge = [comp[str(h)]["mean_mse"]["ridge_ar"] for h in hs]
    idx = next((i for i in range(1, len(hs))
                if gru[i - 1] < ridge[i - 1] and gru[i] >= ridge[i]), None)
    if idx is not None:
        out["crossover_between_ms[0]"] = ms[idx - 1]
        out["crossover_between_ms[1]"] = ms[idx]
    return out


def derive_envelope(art: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for band, hz in ((b, art["bands"][b]["horizons"]) for b in art["bands"]):
        for i, h in enumerate(hz):
            p, m = h["eeg"]["persistence_mse"], h["eeg"]["mse"]
            out[f"bands.{band}.skill_vs_persistence[{i}]"] = (p - m) / p
    return out


def derive_overlap_law(art: dict) -> dict[str, float]:
    synthetic = ("white_noise", "ar1_0.5", "ar1_0.95", "random_walk")
    n = sum(1 for c in art["cells"] if c["signal"] in synthetic)
    return {"n_synthetic_cells": float(n)}


DERIVED = {
    "fig_crossover": derive_crossover,
    "fig_envelope": derive_envelope,
    "fig_overlap_law": derive_overlap_law,
}


def check_numbers(name: str, man: dict, res: Result) -> None:
    art_paths = [man["artifact"]]
    if "main_artifact" in man:
        art_paths.append(man["main_artifact"])

    pool: set[float] = set()
    for rel in art_paths:
        p = ART / rel
        if not p.exists():
            res.add(name, "NUMBERS", False, f"artifact missing: {rel}")
            return
        pool |= artifact_numbers(p)

    with (ART / man["artifact"]).open() as f:
        primary = json.load(f)
    recomputed = DERIVED[name](primary) if name in DERIVED else {}

    leaves = flatten(man)
    unmatched: list[str] = []
    mismatched: list[str] = []
    for key, val in leaves.items():
        head = key.split(".")[0].split("[")[0]
        if head in MANIFEST_META:
            continue
        if key in recomputed:
            if not math.isclose(val, recomputed[key], rel_tol=1e-9,
                                abs_tol=1e-12):
                mismatched.append(
                    f"{key}: figure {val:g} vs recomputed {recomputed[key]:g}")
            continue
        if round(val, 10) in pool:
            continue
        if any(math.isclose(val, c, rel_tol=1e-9, abs_tol=1e-12) for c in pool):
            continue
        unmatched.append(f"{key}={val:g}")

    ok = not unmatched and not mismatched
    if mismatched:
        msg = "DISAGREES WITH ARTIFACT: " + "; ".join(mismatched[:4])
    elif unmatched:
        msg = "not traceable to artifact: " + ", ".join(unmatched[:6])
    else:
        msg = (f"{len(leaves)} values traced "
               f"({len(recomputed)} recomputed independently)")
    res.add(name, "NUMBERS", ok, msg)


# --- check 2: legibility ----------------------------------------------------
def pdf_min_font_pt(pdf: Path) -> float:
    """Smallest text size actually written into the PDF content stream.

    matplotlib emits Tf operators carrying the font size in text space, and the
    text matrix is identity for these figures, so the operand is the printed
    point size. Reading the file is the only way to know what was really
    written, as opposed to what the source asked for.
    """
    reader = PdfReader(str(pdf))
    sizes: list[float] = []
    for page in reader.pages:
        raw = page.get_contents().get_data()
        text = raw.decode("latin-1", errors="ignore")
        for m in re.finditer(r"/[A-Za-z0-9#_+-]+\s+([0-9.]+)\s+Tf", text):
            sizes.append(float(m.group(1)))
    return min(sizes) if sizes else float("nan")


def check_legibility(name: str, man: dict, res: Result) -> None:
    pdf = FIGDIR / f"{name}.pdf"
    if not pdf.exists():
        res.add(name, "LEGIBILITY", False, "no PDF")
        return
    pt = pdf_min_font_pt(pdf)
    if math.isnan(pt):
        res.add(name, "LEGIBILITY", False, "no text found in PDF")
        return
    # Printed size scales with any resize on the page; width check keeps that 1.
    scale = COL_W / man["figure_width_in"]
    printed = pt * scale
    ok = printed >= MIN_PT - 1e-6
    res.add(name, "LEGIBILITY", ok,
            f"min type {printed:.2f} pt at {COL_W} in (floor {MIN_PT})")


# --- checks 3 and 4: geometry ----------------------------------------------
def check_width(name: str, man: dict, res: Result) -> None:
    w = man["figure_width_in"]
    ok = abs(w - COL_W) < 1e-6
    res.add(name, "WIDTH", ok, f"saved {w:.3f} in vs column {COL_W} in")


def check_dpi(name: str, man: dict, res: Result) -> None:
    png = FIGDIR / f"{name}.png"
    if not png.exists():
        res.add(name, "DPI", False, "no PNG")
        return
    px_w = Image.open(png).size[0]
    dpi = px_w / COL_W
    ok = dpi >= MIN_DPI
    res.add(name, "DPI", ok, f"{dpi:.0f} dpi at placement ({px_w} px)")


# --- check 5: retraction scan ----------------------------------------------
def pdf_text(pdf: Path) -> str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages)


def check_retraction(name: str, man: dict, res: Result) -> None:
    hay = pdf_text(FIGDIR / f"{name}.pdf") + json.dumps(man)
    hits = [t for t in RETRACTED if t in hay]
    if not hits:
        res.add(name, "RETRACTION", True, "no retracted values present")
        return
    if name in RETRACTION_CONTEXT:
        res.add(name, "RETRACTION", True,
                f"{', '.join(hits)} present in declared retraction context")
        return
    res.add(name, "RETRACTION", False,
            f"retracted value(s) as live result: {', '.join(hits)}")


# --- check 6: provenance ----------------------------------------------------
def check_provenance(name: str, man: dict, res: Result, prov: dict) -> None:
    rels = [man["artifact"]] + ([man["main_artifact"]]
                                if "main_artifact" in man else [])
    entry = prov.setdefault(name, {})
    msgs, ok = [], True
    for rel in rels:
        p = ART / rel
        if not p.exists():
            ok = False
            msgs.append(f"{rel}: MISSING")
            continue
        h = sha256(p)
        prior = entry.get(rel)
        if prior and prior != h:
            ok = False
            msgs.append(f"{rel}: sha changed since last run")
        else:
            msgs.append(f"{rel}: {h[:12]}")
        entry[rel] = h
    res.add(name, "PROVENANCE", ok, "; ".join(msgs))


# --- check 7: frozen figures ------------------------------------------------
def check_frozen(res: Result, prov: dict) -> None:
    """The five committed hybrid figures must not change.

    git is the authority here rather than a hash file I write myself, because a
    hash file I write cannot detect that I already overwrote the figure before
    recording it.
    """
    rel = [str(p.relative_to(ROOT)) for p in FROZEN]
    if not rel:
        res.add("(frozen)", "NO-REGRESS", False, "no committed figures found")
        return
    out = subprocess.run(["git", "status", "--porcelain", "--"] + rel,
                         cwd=ROOT, capture_output=True, text=True)
    # An empty stdout means "clean" only if git actually answered. Outside a
    # repository git errors and prints nothing, which would otherwise read as
    # a pass and make this check unfalsifiable.
    if out.returncode != 0:
        res.add("(frozen)", "NO-REGRESS", False,
                f"git could not vouch for these files: {out.stderr.strip()[:80]}")
        return
    missing = [p for p in rel if not (ROOT / p).exists()]
    if missing:
        res.add("(frozen)", "NO-REGRESS", False, f"missing: {missing}")
        return
    dirty = [ln for ln in out.stdout.splitlines() if ln.strip()]
    ok = not dirty
    res.add("(frozen)", "NO-REGRESS", ok,
            f"{len(rel)} committed figures unmodified" if ok
            else "modified: " + "; ".join(dirty))
    prov["frozen"] = {p: sha256(ROOT / p) for p in rel}


# --- check 8: dataset honesty ----------------------------------------------
# A figure captioned for a dataset must draw from an artifact of that dataset.
DATASET_TOKENS = {
    "sleep-edf": ("sleep-edf", "sleep_edf", "sleep-cassette", "sleepedf"),
    "bnci": ("bnci",),
}


def check_dataset(name: str, man: dict, res: Result) -> None:
    if "dataset" not in man:
        # A dataset-free figure is allowed only if it makes no dataset claim.
        # The overlap law is arithmetic over many signals, so it is exempt, but
        # it must say so rather than simply omitting the field.
        ok = "law" in man
        res.add(name, "DATASET", ok,
                "no dataset claim (law figure)" if ok
                else "manifest names no dataset")
        return

    claimed = man["dataset"].lower()
    fam = next((k for k, toks in DATASET_TOKENS.items()
                if any(t in claimed for t in toks)), None)
    if fam is None:
        res.add(name, "DATASET", False, f"unrecognised dataset: {man['dataset']}")
        return

    # Evidence comes from the artifact itself, not only its path. An artifact
    # under a neutral directory name would otherwise let any caption stand,
    # which is exactly the Sleep-EDF/BNCI mismatch this check exists to catch.
    hay = (man["artifact"] + " " + man.get("main_artifact", "")).lower()
    for rel in [man["artifact"]] + ([man["main_artifact"]]
                                    if "main_artifact" in man else []):
        p = ART / rel
        if p.exists():
            with p.open() as f:
                a = json.load(f)
            for key in ("dataset", "data", "data_root", "source", "question"):
                v = a.get(key)
                if isinstance(v, str):
                    hay += " " + v.lower()
                elif isinstance(v, dict):
                    hay += " " + json.dumps(v).lower()

    found = {k for k, toks in DATASET_TOKENS.items()
             if any(t in hay for t in toks)}
    if not found:
        res.add(name, "DATASET", False,
                f"claims {fam} but no artifact evidence names any dataset")
        return
    ok = found == {fam}
    res.add(name, "DATASET", ok,
            f"claims {fam}, artifact evidence agrees" if ok
            else f"claims {fam} but artifact evidence names {sorted(found)}")


def main() -> int:
    mans = sorted(FIGDIR.glob("fig_*.json"))
    if not mans:
        print("no figures found")
        return 1

    prov = json.loads(PROV.read_text()) if PROV.exists() else {}
    res = Result()

    for mp in mans:
        name = mp.stem
        man = json.loads(mp.read_text())
        check_numbers(name, man, res)
        check_legibility(name, man, res)
        check_width(name, man, res)
        check_dpi(name, man, res)
        check_retraction(name, man, res)
        check_provenance(name, man, res, prov)
        check_dataset(name, man, res)
    check_frozen(res, prov)

    PROV.write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")

    cur = None
    for fig, check, ok, msg in res.rows:
        if fig != cur:
            print(f"\n{fig}")
            cur = fig
        print(f"  {'PASS' if ok else 'FAIL'}  {check:<11} {msg}")

    n_ok = sum(1 for r in res.rows if r[2])
    print(f"\n{n_ok}/{len(res.rows)} checks pass")
    if res.failed:
        print("WORST: " + res.failed[0][0] + " / " + res.failed[0][1]
              + " -- " + res.failed[0][3])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
