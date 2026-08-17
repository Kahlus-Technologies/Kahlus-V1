"""Draw detected panel boxes over the sources so the split can be eyeballed."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source_wide"
OUT = ROOT / "debug"
COLORS = ["#e11d48", "#0f766e", "#7c3aed", "#ea580c", "#0369a1", "#65a30d"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    report = json.loads((ROOT / "panels.json").read_text())
    for name, info in report.items():
        im = Image.open(SRC / name).convert("RGB")
        d = ImageDraw.Draw(im)
        for i, b in enumerate(info["panels"]):
            c = COLORS[i % len(COLORS)]
            d.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=c, width=8)
            d.rectangle([b["x"], b["y"], b["x"] + 90, b["y"] + 60], fill=c)
            d.text((b["x"] + 12, b["y"] + 20), str(i), fill="white")
        scale = 1100 / im.width
        im = im.resize((1100, int(im.height * scale)))
        im.save(OUT / f"boxes_{name}")
        print(OUT / f"boxes_{name}")


if __name__ == "__main__":
    main()
