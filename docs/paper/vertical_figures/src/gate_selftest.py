"""Negative tests for gate.py.

A guard that cannot be made to fail is not a guard. Each case reintroduces one
specific fault, asserts the matching check FAILs, then restores and asserts the
gate returns to its baseline score. Prints `SELFTEST k/6`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PROV = ROOT / "provenance.json"
FIGS = ROOT / "figures"


def run_gate() -> dict[str, str]:
    out = subprocess.run([sys.executable, str(ROOT / "src" / "gate.py")],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        if line.startswith("CHECK "):
            _, name, val, tgt, status = line.split(maxsplit=4)
            res[name] = status
        elif line.startswith("SCORE "):
            res["_score"] = line.split()[1]
    return res


def main() -> None:
    backup_prov = PROV.read_text()
    backup_figs = ROOT / "_selftest_backup"
    if backup_figs.exists():
        shutil.rmtree(backup_figs)
    shutil.copytree(FIGS, backup_figs)

    base = run_gate()
    passed = 0
    cases: list[tuple[str, str]] = []

    def restore() -> None:
        PROV.write_text(backup_prov)
        shutil.rmtree(FIGS)
        shutil.copytree(backup_figs, FIGS)

    # 1. Redrawn panel: overwrite pixels so they no longer match the source.
    prov = json.loads(backup_prov)
    r = prov[0]
    im = Image.open(FIGS / r["figure"])
    q = r["panels"][0]["placed"]
    im.paste((255, 0, 0), (q["x"], q["y"], q["x"] + 60, q["y"] + 60))
    im.save(FIGS / r["figure"])
    cases.append(("redrawn_panel", "n_panels_pixel_matched"))
    got = run_gate()
    passed += got.get("n_panels_pixel_matched") == "FAIL"
    restore()

    # 2. Upscaled panel.
    prov = json.loads(backup_prov)
    prov[0]["panels"][0]["scale"] = 0.5
    PROV.write_text(json.dumps(prov))
    got = run_gate()
    passed += got.get("n_upscaled_panels") == "FAIL"
    cases.append(("upscaled_panel", "n_upscaled_panels"))
    restore()

    # 3. Overlapping panels.
    prov = json.loads(backup_prov)
    prov[0]["panels"][1]["placed"] = dict(prov[0]["panels"][0]["placed"])
    PROV.write_text(json.dumps(prov))
    got = run_gate()
    passed += got.get("n_panel_overlaps") == "FAIL"
    cases.append(("overlap", "n_panel_overlaps"))
    restore()

    # 4. Edge clipping / bleed.
    prov = json.loads(backup_prov)
    prov[0]["panels"][0]["placed"]["x"] = 0
    PROV.write_text(json.dumps(prov))
    got = run_gate()
    passed += got.get("n_edge_clipped") == "FAIL"
    cases.append(("edge_clip", "n_edge_clipped"))
    restore()

    # 5. Side-by-side panels break the single-column stack.
    prov = json.loads(backup_prov)
    a = prov[0]["panels"][0]["placed"]
    prov[0]["panels"][1]["placed"]["y"] = a["y"]
    PROV.write_text(json.dumps(prov))
    got = run_gate()
    passed += got.get("n_panels_single_column") == "FAIL"
    cases.append(("side_by_side", "n_panels_single_column"))
    restore()

    # 6. Dropped figure shrinks coverage below the fixed denominator.
    prov = [r for r in json.loads(backup_prov) if len(r["panels"]) > 1][:-1]
    PROV.write_text(json.dumps(prov))
    got = run_gate()
    passed += got.get("n_figures_rebuilt") == "FAIL"
    cases.append(("dropped_figure", "n_figures_rebuilt"))
    restore()

    after = run_gate()
    shutil.rmtree(backup_figs)

    for name, check in cases:
        print(f"CASE {name} -> {check}")
    print(f"SELFTEST {passed}/6")
    print(f"RESTORED score {after.get('_score')} (baseline {base.get('_score')})")
    if after.get("_score") != base.get("_score"):
        raise SystemExit("restore failed: gate score changed")


if __name__ == "__main__":
    main()
