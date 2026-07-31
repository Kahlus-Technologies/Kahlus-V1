"""Compose slim vertical (IEEE half-column) figures from wide source artwork.

Every panel placed here is a byte-exact rectangular crop of a 2600px-wide
source image. Nothing is redrawn, nothing is resampled, nothing is upscaled.
The only pixels this module authors are chrome: background, panel frames,
lettered band labels, and connector arrows.

WHY THIS HELPS, AND ITS CEILING
-------------------------------
Printed text size obeys

    printed_pt = glyph_px / canvas_px * 72 * placed_width_in

Note what is absent: the number of rows. Stacking panels vertically does not
enlarge type by itself, because a taller figure is still placed at the same
column width. The single lever available to a crop-only pipeline is the
*canvas width*. A 2x2 grid forces a 2600px canvas; one panel per row lets the
canvas shrink to a single panel (~1200px), which doubles printed type at the
same placement. That is the entire, real gain of this rewrite: about 2.0x for
the 4-panel figures and about 2.9x for the 3-panel ones.

It is not enough to reach 8pt. Hitting 8pt at 3.25in from a 22px glyph needs a
canvas under ~644px, and the narrowest panel here is 755px. So 8pt is
unreachable by cropping alone; it requires re-rendering the artwork with larger
fonts. This module goes as far as cropping honestly can, and `gate.py` reports
the shortfall as a number rather than hiding it.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "source_wide"
OUT = ROOT / "figures"

COL_IN = 3.25          # IEEE two-column half-column width (maximum)
MIN_DPI = 300.0        # print floor; narrow figures are placed under COL_IN to hold it
PAD = 34               # canvas margin around panels
GAP = 128              # vertical space between panels, holds the arrow
LABEL_H = 52           # lettered band above each panel
FRAME = 3              # panel border weight
BG = "white"
INK = "#111111"
FRAME_C = "#444444"
ARROW_C = "#777777"


def _font(px: int) -> ImageFont.FreeTypeFont:
    for cand in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, px)
            except OSError:
                continue
    return ImageFont.load_default()


def compose(name: str, panels: list[dict], out_path: Path) -> dict:
    """Stack one figure's panels vertically. Returns a provenance record."""
    src = Image.open(SRC / name).convert("RGB")
    crops = [src.crop((p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"])) for p in panels]

    # Canvas is exactly as wide as the widest panel. Narrower panels are
    # centred, never stretched, since scaling up would resample the artwork.
    inner = max(c.width for c in crops)
    cw = inner + 2 * PAD
    ch = PAD + sum(LABEL_H + c.height for c in crops) + GAP * (len(crops) - 1) + PAD

    canvas = Image.new("RGB", (cw, ch), BG)
    d = ImageDraw.Draw(canvas)
    f = _font(LABEL_H - 14)

    placed = []
    y = PAD
    for i, (c, p) in enumerate(zip(crops, panels)):
        letter = chr(ord("a") + i)
        d.text((PAD, y + 4), f"({letter})", fill=INK, font=f)
        y += LABEL_H

        x = PAD + (inner - c.width) // 2
        canvas.paste(c, (x, y))  # 1:1 paste, no resize
        d.rectangle([x - FRAME, y - FRAME, x + c.width + FRAME - 1,
                     y + c.height + FRAME - 1], outline=FRAME_C, width=FRAME)
        placed.append({
            "letter": letter,
            "source": name,
            "crop": {"x": p["x"], "y": p["y"], "w": p["w"], "h": p["h"]},
            "placed": {"x": x, "y": y, "w": c.width, "h": c.height},
            "scale": 1.0,
        })
        y += c.height

        if i < len(crops) - 1:  # connector arrow into the next panel
            cx, a0, a1 = cw // 2, y + 30, y + GAP - 30
            d.line([cx, a0, cx, a1], fill=ARROW_C, width=4)
            d.polygon([(cx - 13, a1 - 15), (cx + 13, a1 - 15), (cx, a1)], fill=ARROW_C)
            y += GAP

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Place at the column width, but narrow enough to hold the print floor.
    # The 3-panel sources are 1x3 strips, so a single panel is only ~850px;
    # at the full 3.25in that lands near 270dpi. Placing such a figure at
    # ~2.9in keeps it above 300dpi. This shrinks the printed figure slightly
    # rather than resampling it, so the artwork stays byte-exact.
    placed_in = min(COL_IN, cw / MIN_DPI)
    dpi = round(cw / placed_in)
    canvas.save(out_path, dpi=(dpi, dpi))

    return {
        "figure": out_path.name,
        "source": name,
        "canvas": {"w": cw, "h": ch},
        "aspect": round(ch / cw, 3),
        "placed_width_in": round(placed_in, 3),
        "effective_dpi": round(cw / placed_in, 1),
        "panels": placed,
    }


def main() -> None:
    panels = json.loads((ROOT / "panels.json").read_text())
    prov = []
    for name, info in panels.items():
        stem = Path(name).stem
        rec = compose(name, info["panels"], OUT / f"{stem}_vertical.png")
        prov.append(rec)
        print(f"{rec['figure']:26s} {rec['canvas']['w']}x{rec['canvas']['h']} "
              f"aspect {rec['aspect']:.2f}  {rec['effective_dpi']:.0f}dpi "
              f"@{rec['placed_width_in']:.2f}in  {len(rec['panels'])} panels")
    (ROOT / "provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
