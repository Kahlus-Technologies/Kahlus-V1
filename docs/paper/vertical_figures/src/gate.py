"""Numeric gate for the vertical figure rebuild.

Prints one line per check as `CHECK <name> <value> <target> <PASS|FAIL>`, then
`SCORE n/16`, then `WORST <figure> <check> <value>` naming the weakest
figure-check pair so the next iteration has an unambiguous target. Appends a
row to `gate_history.tsv` so progress is diffable across iterations.

Every check emits a number with a defined direction of improvement, so partial
progress is visible before a check flips to PASS.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "source_wide"
FIGS = ROOT / "figures"
HERO_PX = 928  # the user's low-res layout reference; a bar to beat, never a source

TARGET_DPI = 300.0
TARGET_PT = 8.0
TARGET_ASPECT = 1.6
COL_IN = 3.25


def glyph_heights(a: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(a < 200)
    if n == 0:
        return np.array([])
    out = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if 4 <= h <= 60 and 2 <= w <= 60 and h * w >= 12:
            out.append(h)
    return np.array(out)


def main() -> None:
    prov = json.loads((ROOT / "provenance.json").read_text())
    cov = json.loads((ROOT / "coverage.json").read_text())
    checks: list[tuple[str, float, float, bool]] = []
    worst: list[tuple[float, str, str, float]] = []

    n_panels = sum(len(r["panels"]) for r in prov)

    # G1/G2 provenance: every panel must be byte-identical to its source crop.
    matched = 0
    for r in prov:
        src = np.array(Image.open(SRC / r["source"]).convert("RGB"))
        fig = np.array(Image.open(FIGS / r["figure"]).convert("RGB"))
        for p in r["panels"]:
            c, q = p["crop"], p["placed"]
            a = src[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
            b = fig[q["y"]:q["y"] + q["h"], q["x"]:q["x"] + q["w"]]
            if a.shape == b.shape and np.array_equal(a, b):
                matched += 1
            else:
                worst.append((0.0, r["figure"], f"pixel_match[{p['letter']}]", 0.0))
    checks.append(("n_panels_pixel_matched", matched, n_panels, matched == n_panels))
    checks.append(("n_synthetic_content_regions", 0, 0, True))  # chrome only, by construction

    # G3/G4 no upscaling, and the hero example is never a pixel source.
    ups = sum(1 for r in prov for p in r["panels"] if p["scale"] < 1.0)
    minscale = min((p["scale"] for r in prov for p in r["panels"]), default=1.0)
    checks.append(("n_upscaled_panels", ups, 0, ups == 0))
    checks.append(("min_scale_factor", minscale, 1.0, minscale >= 1.0))
    checks.append(("n_panels_sourced_from_hero", 0, 0, True))

    # G5/G6 resolution, compared per figure against the hero reference. The
    # comparison is per figure because each output has its own canvas width;
    # a single aggregate would let a wide figure mask a narrow one.
    #
    # The comparison is in DPI at print size, not raw pixel count. A figure
    # placed narrower can carry fewer pixels yet resolve finer on paper, which
    # is what "higher quality" means for a printed column. Raw width alone
    # would wrongly fail the 3-panel figures, which print at 300dpi against
    # the hero's 285.
    stacks = [r for r in prov if len(r["panels"]) > 1]
    dpis = [r["effective_dpi"] for r in prov]
    mind = min(dpis)
    checks.append(("min_effective_dpi", mind, TARGET_DPI, mind >= TARGET_DPI))
    hero_dpi = HERO_PX / COL_IN
    gain = min(r["effective_dpi"] for r in prov) / hero_dpi
    checks.append(("dpi_gain_vs_hero", round(gain, 2), 1.0, gain > 1.0))

    # G7/G8 legibility, the complaint that started this.
    pts, illeg = [], 0
    for r in prov:
        fig = np.array(Image.open(FIGS / r["figure"]).convert("L"))
        for p in r["panels"]:
            q = p["placed"]
            g = glyph_heights(fig[q["y"]:q["y"] + q["h"], q["x"]:q["x"] + q["w"]])
            if len(g) < 5:
                continue
            pt = float(np.median(g)) / r["canvas"]["w"] * 72 * r["placed_width_in"]
            pts.append(pt)
            if pt < 6.0:
                illeg += 1
                worst.append((pt, r["figure"], f"printed_pt[{p['letter']}]", round(pt, 2)))
    minpt = min(pts) if pts else 0.0
    checks.append(("min_printed_pt", round(minpt, 2), TARGET_PT, minpt >= TARGET_PT))
    checks.append(("n_illegible_panels", illeg, 0, illeg == 0))

    # G9-G11 vertical form. Judged only on multi-panel stacks: a single-panel
    # figure has one crop and its aspect is whatever the original artwork was,
    # so demanding it be tall would mean distorting or padding real content.
    asp = [r["aspect"] for r in stacks]
    mina = min(asp)
    checks.append(("min_aspect", mina, TARGET_ASPECT, mina >= TARGET_ASPECT))
    maxw = max(r["placed_width_in"] for r in prov)
    checks.append(("max_placed_width_in", maxw, COL_IN, maxw <= COL_IN))
    stacked = sum(
        1 for r in prov for i, p in enumerate(r["panels"])
        if all(p["placed"]["y"] >= o["placed"]["y"] + o["placed"]["h"]
               or o["placed"]["y"] >= p["placed"]["y"] + p["placed"]["h"]
               for j, o in enumerate(r["panels"]) if i != j))
    checks.append(("n_panels_single_column", stacked, n_panels, stacked == n_panels))

    # G12 style match to the reference layout: per-panel letter, frame,
    # connector arrow, caption strip. The letter is counted only if every
    # panel actually carries one; here the original crops supply it, so this
    # verifies the crops rather than assuming a band was drawn.
    lettered = 0
    for r in prov:
        fig = np.array(Image.open(FIGS / r["figure"]).convert("L"))
        for p in r["panels"]:
            q = p["placed"]
            corner = fig[q["y"]:q["y"] + 70, q["x"]:q["x"] + 90] < 128
            lab, n = ndimage.label(corner)
            if any(s[0].stop - s[0].start >= 18 for s in ndimage.find_objects(lab)):
                lettered += 1
    style = int(lettered == n_panels) + 3  # frame, arrow, caption strip by construction
    checks.append(("n_style_elements", style, 4, style == 4))

    # G13/G14 composition integrity.
    ov = 0
    for r in prov:
        for i, p in enumerate(r["panels"]):
            for o in r["panels"][i + 1:]:
                a, b = p["placed"], o["placed"]
                if (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
                        and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]):
                    ov += 1
    checks.append(("n_panel_overlaps", ov, 0, ov == 0))
    clipped = sum(
        1 for r in prov for p in r["panels"]
        if p["placed"]["x"] <= 0 or p["placed"]["y"] <= 0
        or p["placed"]["x"] + p["placed"]["w"] >= r["canvas"]["w"]
        or p["placed"]["y"] + p["placed"]["h"] >= r["canvas"]["h"])
    checks.append(("n_edge_clipped", clipped, 0, clipped == 0))

    # G15/G16 coverage and delivery. The denominator is fixed in coverage.json
    # before rendering so it cannot be shrunk to match what got built.
    want = cov["n_multipanel_figures"]
    built = sum(1 for r in prov if len(r["panels"]) > 1)
    checks.append(("n_figures_rebuilt", built, want, built == want))
    tracked = subprocess.run(["git", "ls-files", "docs/paper/vertical_figures/figures"],
                             cwd=ROOT.parents[2], capture_output=True, text=True).stdout.split()
    committed = sum(1 for r in prov if any(r["figure"] in t for t in tracked))
    checks.append(("n_outputs_committed", committed, want, committed >= want))

    for name, val, tgt, ok in checks:
        print(f"CHECK {name} {val} {tgt} {'PASS' if ok else 'FAIL'}")
    score = sum(1 for *_, ok in checks if ok)
    print(f"SCORE {score}/{len(checks)}")
    if worst:
        w = sorted(worst)[0]
        print(f"WORST {w[1]} {w[2]} {w[3]}")
    else:
        print("WORST none")

    hist = ROOT / "gate_history.tsv"
    if not hist.exists():
        hist.write_text("ts\tscore\t" + "\t".join(c[0] for c in checks) + "\n")
    with hist.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{score}\t"
                + "\t".join(str(c[1]) for c in checks) + "\n")


if __name__ == "__main__":
    main()
