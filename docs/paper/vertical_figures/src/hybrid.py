"""Hybrid figure composition: figurelabs artwork + precise data panels.

The two sources do different jobs. The figurelabs renders explain the *process*
(a scalp EEG cap, a latent field, a compiler pipeline) and read well at a
glance. The matplotlib panels carry the *evidence* (exact axes, tick labels,
numeric annotations). A paper figure wants both: an artistic opener that
orients the reader, then measured panels underneath that support the claim.

Every region placed here, from either source, is a byte-exact rectangular crop.
Nothing is redrawn or resampled. `gate.py` template-matches all of them back to
their origin files.

ASSET REGISTRY
--------------
Crop boxes were located by probing the source and visually confirming each
region. They are recorded as data rather than buried in the layout code so a
reader can audit "which pixels came from where" without following control flow.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "source_wide"
ART = ROOT / "source_art"
OUT = ROOT / "figures"

COL_IN = 3.25
MIN_DPI = 300.0
PAD = 34
GAP = 120
FRAME = 3
BG = "white"
FRAME_C = "#444444"
ARROW_C = "#777777"
CAP_C = "#222222"

# Figurelabs regions, located by scanning the visual abstract for horizontal
# whitespace bands rather than by eyeballing coordinates. A first attempt
# guessed boxes and produced clipped headings plus a stray arc from a
# neighbouring panel; the scan found the true content boundaries
# (y=88..410 and y=419..765). Each box starts below its band's bold heading so
# the crop carries artwork, not a half-line of section title.
ASSETS: dict[str, dict] = {
    "eeg_windowing": {
        "file": "visual_abstract.png",
        "box": (0, 120, 928, 405),
        "why": "Scalp-cap render with electrode array feeding a labelled input "
               "window. Replaces the plain circle-with-dots schematic and the "
               "trace panel beside it in one self-contained image.",
    },
    "latent_field": {
        "file": "visual_abstract.png",
        "box": (0, 450, 928, 762),
        "why": "Neural Field Compiler band: the latent field with its hidden "
               "state slices, the paper's central construct.",
    },
}


def _font(px: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["Arial Bold.ttf"] if bold else ["Arial.ttf", "Helvetica.ttc"]
    for n in names + ["Arial.ttf"]:
        p = Path("/System/Library/Fonts/Supplemental") / n
        if p.exists():
            try:
                return ImageFont.truetype(str(p), px)
            except OSError:
                pass
    return ImageFont.load_default()


def load_asset(key: str) -> Image.Image:
    a = ASSETS[key]
    return Image.open(ART / a["file"]).convert("RGB").crop(a["box"])



def compose(name: str, rows: list[dict], out_path: Path, caption: str) -> dict:
    """Stack heterogeneous rows (art or data) into one column figure."""
    imgs = []
    for r in rows:
        if r["kind"] == "art":
            imgs.append(load_asset(r["asset"]))
        else:
            src = Image.open(SRC / r["source"]).convert("RGB")
            c = r["crop"]
            imgs.append(src.crop((c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"])))

    inner = max(i.width for i in imgs)
    cw = inner + 2 * PAD
    cap_h = 46 if caption else 0
    ch = PAD + sum(i.height for i in imgs) + GAP * (len(imgs) - 1) + PAD + cap_h

    canvas = Image.new("RGB", (cw, ch), BG)
    d = ImageDraw.Draw(canvas)

    placed, y = [], PAD
    for i, (im, r) in enumerate(zip(imgs, rows)):
        x = PAD + (inner - im.width) // 2
        canvas.paste(im, (x, y))
        d.rectangle([x - FRAME, y - FRAME, x + im.width + FRAME - 1,
                     y + im.height + FRAME - 1], outline=FRAME_C, width=FRAME)
        placed.append({
            "letter": chr(ord("a") + i),
            "kind": r["kind"],
            "origin": r.get("asset") or r["source"],
            "crop": (dict(zip(("x0", "y0", "x1", "y1"), ASSETS[r["asset"]]["box"]))
                     if r["kind"] == "art" else r["crop"]),
            "placed": {"x": x, "y": y, "w": im.width, "h": im.height},
            "scale": 1.0,
        })
        y += im.height
        if i < len(imgs) - 1:
            cx, a0, a1 = cw // 2, y + 26, y + GAP - 26
            d.line([cx, a0, cx, a1], fill=ARROW_C, width=4)
            d.polygon([(cx - 13, a1 - 15), (cx + 13, a1 - 15), (cx, a1)], fill=ARROW_C)
            y += GAP

    if caption:
        d.text((PAD, ch - cap_h + 6), caption, fill=CAP_C, font=_font(26))

    placed_in = min(COL_IN, cw / MIN_DPI)
    dpi = round(cw / placed_in)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))

    return {
        "figure": out_path.name,
        "source": name,
        "canvas": {"w": cw, "h": ch},
        "aspect": round(ch / cw, 3),
        "placed_width_in": round(placed_in, 3),
        "effective_dpi": round(cw / placed_in, 1),
        "caption": caption,
        "panels": placed,
    }
