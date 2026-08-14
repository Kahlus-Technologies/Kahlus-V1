"""Convert flat raster art into resolution-independent vector PDF.

Why this exists
---------------
The art generator caps output at about 1.5 megapixels. Placed at the full
IEEE column measure of 3.5 in, that is roughly 300-400 dpi against a 600 dpi
print floor. Asking the generator for more does not work; the cap is hard.

Upscaling was measured and rejected. Round-tripping a known panel through the
required 1.44x factor retained only 82.7% of edge contrast, meaning visibly
softer strokes. Softness is the exact complaint being fixed, so inventing
pixels is not an option.

Tracing is different in kind. This art is flat: on a representative panel,
three colours account for 94% of pixels and the whole image quantises to a
small palette with hard edges. That is a rendered vector drawing which was
rasterised. Recovering the regions as paths does not invent detail, it
reconstructs the shapes the renderer drew, and the result prints at any size.

Method
------
1. Quantise to a small palette, since the source is flat by construction.
2. For each colour, build a binary mask and trace it with potrace to a path.
3. Stack the paths back to front into one SVG, then convert to PDF.
4. Verify by rasterising the result and comparing against the source. A trace
   that drifts from the original is a failure, not an improvement, so the
   comparison gates the output.

Text is traced as filled outlines rather than live glyphs. For figure art that
is correct: the shapes are preserved exactly and no font needs to embed or
substitute at the publisher.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ART = BASE / "art"
VEC = BASE / "art_vector"
VEC.mkdir(parents=True, exist_ok=True)

# Below this share of the image a colour is antialiasing, not a design element.
MIN_REGION = 0.0008
MAX_COLOURS = 18
# Pale tints sit within a few steps of white and must not be quantised
# into the background, or filled boxes render hollow.
QUANT = 8
SUPERSAMPLE = 4          # trace at 4x for sub-pixel accurate path placement
# Quantising to a grid of QUANT scatters antialiased white pixels across bins
# one step off pure white. Those bins are not junk: a pale box fill lands in
# the same bin as the haze, so dropping the colour empties real boxes. What
# separates them is size, not hue. A fill is one large connected region; the
# haze is thousands of specks. Near-white masks therefore keep only their
# large components. Dark colours are left alone, since small dark parts are
# real (dashes, dots on an i, thin arrowheads).
NEAR_WHITE = 40.0        # RGB distance from background, 0..255
MIN_BLOB = 0.00150       # fraction of image area a pale region must cover


def _flatten(path: Path) -> np.ndarray:
    """Load as RGB float in 0..1, compositing any alpha onto white."""
    im = Image.open(path)
    a = np.asarray(im, dtype=float) / 255.0
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    if a.shape[2] == 4:
        al = a[..., 3:4]
        a = a[..., :3] * al + (1.0 - al)
    return a[..., :3]


def _palette(a: np.ndarray) -> list[np.ndarray]:
    """Recover the design palette, ordered by coverage.

    Antialiased edge pixels are blends and must not become their own regions,
    so colours are snapped to a coarse grid and rare bins dropped.
    """
    q = (a * 255).astype(np.uint8) // QUANT * QUANT
    cols, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    keep = counts / counts.sum() >= MIN_REGION
    cols, counts = cols[keep], counts[keep]
    order = np.argsort(-counts)
    return [cols[i].astype(float) / 255.0 for i in order[:MAX_COLOURS]]


def _assign(a: np.ndarray, palette: list[np.ndarray]) -> np.ndarray:
    """Map every pixel to its nearest palette entry."""
    flat = a.reshape(-1, 3)
    pal = np.stack(palette)
    d = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).reshape(a.shape[:2])


def _drop_specks(mask: np.ndarray) -> np.ndarray:
    """Keep only connected regions large enough to be a drawn shape."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    keep = areas >= MIN_BLOB * mask.size
    return keep[lab]


def _trace_mask(mask: np.ndarray, scale: int) -> str | None:
    """Trace one binary mask to SVG path data via potrace."""
    if not mask.any():
        return None
    big = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
    with tempfile.TemporaryDirectory() as td:
        pbm = Path(td) / "m.pbm"
        svg = Path(td) / "m.svg"
        Image.fromarray(np.where(big, 0, 255).astype(np.uint8)).save(pbm)
        r = subprocess.run(
            ["potrace", str(pbm), "-s", "-o", str(svg),
             "--turdsize", "2", "--alphamax", "1.0", "--opttolerance", "0.2"],
            capture_output=True)
        if r.returncode != 0 or not svg.exists():
            return None
        return svg.read_text()


def _paths_from(svg_text: str) -> tuple[str, str]:
    """Return (transform, paths) from a potrace SVG.

    potrace emits coordinates at ten times scale and corrects for it with a
    transform on the wrapping group, along with a y-flip for its bottom-left
    origin. Reusing that exact transform is essential; substituting a
    hand-written one puts every path ten times off-canvas.
    """
    g = svg_text.find("<g transform=")
    tstart = svg_text.find('"', g) + 1
    tend = svg_text.find('"', tstart)
    transform = svg_text[tstart:tend]

    out, i = [], 0
    while True:
        i = svg_text.find("<path", i)
        if i < 0:
            break
        j = svg_text.find("/>", i)
        out.append(svg_text[i:j + 2])
        i = j
    return transform, "\n".join(out)


def vectorize(name: str, verbose: bool = True) -> Path:
    src = ART / name
    a = _flatten(src)
    h, w = a.shape[:2]
    palette = _palette(a)
    idx = _assign(a, palette)

    # Only true white is treated as background. Choosing "largest region"
    # instead would swallow a pale box fill that happens to dominate the
    # frame, and the box would print hollow.
    lum = np.array([c.mean() for c in palette])
    bg = int(np.argmax(lum))
    if lum[bg] < 0.97:
        bg = -1

    # Paint large pale areas before small dark ones so outlines and text sit
    # on top of their fills rather than under them.
    order = sorted(range(len(palette)),
                   key=lambda k: -float(np.asarray(palette[k]).mean()))

    layers = []
    for k in order:
        col = palette[k]
        if k == bg:
            continue
        mask = idx == k
        if bg >= 0 and float(np.linalg.norm(
                (col - palette[bg]) * 255.0)) < NEAR_WHITE:
            mask = _drop_specks(mask)
        svg = _trace_mask(mask, SUPERSAMPLE)
        if not svg:
            continue
        transform, paths = _paths_from(svg)
        if not paths.strip():
            continue
        hexc = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in col)
        layers.append((k, hexc, transform, paths))

    # Masks were traced at SUPERSAMPLE resolution, so the user-space canvas is
    # that much larger; the viewBox maps it back to the original dimensions.
    sw, sh = w * SUPERSAMPLE, h * SUPERSAMPLE
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" '
        f'viewBox="0 0 {sw} {sh}">',
        f'<rect x="0" y="0" width="{sw}" height="{sh}" fill="#ffffff"/>',
    ]
    for _, hexc, transform, paths in layers:
        parts.append(f'<g transform="{transform}" fill="{hexc}" stroke="none">')
        parts.append(paths.replace(' fill="#000000"', ''))
        parts.append('</g>')
    parts.append('</svg>')

    out_svg = VEC / (Path(name).stem + ".svg")
    out_svg.write_text("\n".join(parts))

    out_pdf = VEC / (Path(name).stem + ".pdf")
    r = subprocess.run(["magick", str(out_svg), "-density", "1200", str(out_pdf)],
                       capture_output=True)
    if r.returncode != 0:
        out_pdf = out_svg

    if verbose:
        print(f"  {name:22s} {len(layers):2d} colour layers -> {out_pdf.name}")
    return out_pdf


def verify(name: str) -> float:
    """Rasterise the trace and compare with the source.

    Returns mean absolute error in 0..255. A good trace of flat art sits in
    the low single digits; anything higher means the palette or region
    threshold dropped something that mattered.
    """
    src = _flatten(ART / name)
    h, w = src.shape[:2]
    svg = VEC / (Path(name).stem + ".svg")
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "r.png"
        subprocess.run(["magick", "-background", "white", "-density", "300",
                        str(svg), "-resize", f"{w}x{h}!", str(png)],
                       capture_output=True)
        if not png.exists():
            return float("nan")
        got = np.asarray(Image.open(png).convert("RGB"), dtype=float) / 255.0
    return float(np.abs(src - got).mean() * 255)


def main() -> int:
    names = sys.argv[1:] or [p.name for p in sorted(ART.glob("*.png"))]
    print("vectorizing (raster art -> resolution-independent paths)")
    worst = 0.0
    for n in names:
        vectorize(n)
        e = verify(n)
        worst = max(worst, 0 if e != e else e)
        flag = "ok" if e == e and e < 8 else "CHECK"
        print(f"    verify MAE {e:5.2f}/255  {flag}")
    print(f"\nworst MAE {worst:.2f}/255")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
