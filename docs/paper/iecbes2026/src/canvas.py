"""A small absolute-placement canvas for composing figure panels.

Why this exists
---------------
The previous compositor asked matplotlib to lay out panels via gridspec. That
fights the tool in two ways that both produced visible defects:

  * ``imshow`` forces an equal aspect, so an axes silently shrinks to fit its
    image and re-centres. Anything anchored to that axes (a panel letter)
    lands somewhere different on every row.
  * ``savefig(bbox_inches="tight")`` grows the canvas outward to fit tick
    labels. It grows on whichever side needs it, so a figure authored centred
    comes out off-centre. Measured on Fig 3: art at 57.1% of width against the
    bar chart at 50.0%.

Both are matplotlib doing exactly what it is designed to do. The task is not a
plot layout task, it is a page layout task: put this drawing at this position
at this size. So this module does that directly, on top of ``svgutils``, which
is the established tool for compositing publication panels and keeps
everything vector on the way to PDF.

What it guarantees
------------------
1. Placement is absolute and in inches. Nothing re-centres behind your back.
2. Panels are placed by their *ink* box, not their canvas box. The generated
   art carries asymmetric white padding (head_montage is 36% ink by width), so
   canvas-centring a panel is not the same as centring the drawing, which is
   the defect that reads as "the top image is a bit to the right".
3. ``centre_line`` is one number for the whole figure. Every panel that asks
   to be centred shares it, by construction rather than by coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import svgutils.transform as sg
from PIL import Image

PT_PER_IN = 72.0


def ink_box(png: Path, thresh: float = 0.975) -> tuple[float, float, float, float]:
    """Fractional (x0, y0, x1, y1) of the drawing inside its canvas.

    Read off a raster so nothing about the vector's structure (a backing rect,
    a clip path, a stray transform) can fool the measurement.
    """
    im = Image.open(png)
    if im.mode in ("RGBA", "LA"):
        # A transparent panel has no white to threshold against; use alpha.
        alpha = np.asarray(im.split()[-1], dtype=float) / 255.0
        ink = alpha > 0.02
    else:
        a = np.asarray(im.convert("L"), dtype=float) / 255.0
        ink = a < thresh
    if not ink.any():
        raise ValueError(f"{png.name}: no ink found")
    cols = np.where(ink.any(axis=0))[0]
    rows = np.where(ink.any(axis=1))[0]
    h, w = ink.shape
    return (cols.min() / w, rows.min() / h, (cols.max() + 1) / w, (rows.max() + 1) / h)


def ink_box_svg(svg: Path, dpi: float = 150.0) -> tuple[float, float, float, float]:
    """Same measurement for a panel that has no raster twin.

    Renders to a temporary PNG on transparent ground. Used for the matplotlib
    data panels, which are generated fresh each build.
    """
    import tempfile

    from render import svg_to_png_transparent

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "probe.png"
        svg_to_png_transparent(svg, p, dpi=dpi)
        return ink_box(p)


@dataclass
class Placed:
    """One panel on the canvas, in inches, measured by its ink."""

    name: str
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2


class Canvas:
    """A fixed-size page you place panels onto."""

    def __init__(self, width_in: float, height_in: float):
        self.w = width_in
        self.h = height_in
        self._els: list[sg.FigureElement] = []
        self.placed: list[Placed] = []

    @property
    def centre_line(self) -> float:
        return self.w / 2

    def add_svg(self, path: Path, *, y: float, width: float | None = None,
                cx: float | None = None, x: float | None = None,
                png: Path | None = None, name: str | None = None) -> Placed:
        """Place a vector panel so its *ink* occupies the given box.

        ``width`` is the width of the drawing, not of its canvas. Padding is
        translated away, so two panels asked for the same width print the same
        width regardless of how much whitespace each generator baked in.
        """
        png = png or path.with_suffix(".png")
        if not png.exists():
            alt = path.parent.parent / "art" / (path.stem + ".png")
            png = alt if alt.exists() else None
        # No raster twin (the matplotlib data panels are generated fresh each
        # build), so measure the vector by rendering it.
        fx0, fy0, fx1, fy1 = ink_box(png) if png else ink_box_svg(path)

        fig = sg.fromfile(str(path))
        cw, ch = (float(s.replace("pt", "")) for s in fig.get_size())

        # The traced art declares width/height in pt but draws in viewBox user
        # units, and potrace emits a viewBox 4x the pt size. svgutils' getroot()
        # returns the children *without* the outer <svg>, so that viewBox
        # mapping is dropped and the drawing arrives 4x too large. Recover the
        # factor and fold it into the placement scale.
        #
        # Caught by a known-answer test: a panel asked for at 2.00in wide
        # rendered with nothing inside the band it was placed in.
        vb = fig.root.get("viewBox")
        ux, uy = cw, ch
        if vb:
            _, _, vw, vh = (float(v) for v in vb.replace(",", " ").split())
            ux, uy = vw, vh
        unit = ux / cw          # user units per pt

        ink_w_pt = cw * (fx1 - fx0)
        ink_h_pt = ch * (fy1 - fy0)
        ink_w_in = ink_w_pt / PT_PER_IN
        ink_h_in = ink_h_pt / PT_PER_IN

        width = width or ink_w_in
        s = (width / ink_w_in) / unit        # user units -> inches on the page
        h = ink_h_in * (width / ink_w_in)

        if cx is None and x is None:
            cx = self.centre_line
        left = x if x is not None else cx - width / 2

        root = fig.getroot()
        # The traced art paints an opaque white rect over its whole canvas,
        # which would erase anything already placed underneath. Drop it; the
        # page supplies the white.
        self._strip_backing_rect(root, ux, uy)
        # One transform, applied as translate(...) scale(...): scale about the
        # origin, then shift so the *drawing's* top-left corner lands on
        # (left, y). Subtracting the scaled padding is what makes a panel
        # place by its ink instead of by its canvas.
        root.moveto((left * PT_PER_IN) - (cw * fx0 * s * unit),
                    (y * PT_PER_IN) - (ch * fy0 * s * unit),
                    scale_x=s)
        self._els.append(root)

        p = Placed(name or path.stem, left, y, width, h)
        self.placed.append(p)
        return p

    @staticmethod
    def _strip_backing_rect(root, ux: float, uy: float, tol: float = 0.02) -> int:
        """Remove a full-canvas opaque rect from a traced panel.

        potrace-derived art opens with a white rect covering the whole canvas.
        Placed on a page that rect is not background, it is paint, and it hides
        whatever was placed before it. Only rects that fill (near enough) the
        entire canvas are removed, so a legitimate white box inside a diagram
        survives.
        """
        killed = 0
        for el in list(root.root.iter()):
            if not el.tag.endswith("rect"):
                continue
            try:
                w = float(el.get("width", 0))
                h = float(el.get("height", 0))
                x = float(el.get("x", 0))
                y = float(el.get("y", 0))
            except (TypeError, ValueError):
                continue
            fill = (el.get("fill") or "").lower()
            covers = (w >= ux * (1 - tol) and h >= uy * (1 - tol)
                      and abs(x) <= ux * tol and abs(y) <= uy * tol)
            if covers and fill in ("#ffffff", "#fff", "white"):
                el.getparent().remove(el)
                killed += 1
        return killed

    def add_text(self, text: str, x: float, y: float, *, size: float = 9.0,
                 weight: str = "bold", colour: str = "#1a1a1a",
                 family: str = "Helvetica", anchor: str = "start") -> None:
        """Text placed by baseline, in inches."""
        t = sg.TextElement(x * PT_PER_IN, y * PT_PER_IN, text,
                           size=size, weight=weight, color=colour,
                           font=family, anchor=anchor)
        self._els.append(t)

    def save_svg(self, out: Path) -> Path:
        fig = sg.SVGFigure()
        fig.set_size((f"{self.w * PT_PER_IN}pt", f"{self.h * PT_PER_IN}pt"))
        fig.root.set("viewBox", f"0 0 {self.w * PT_PER_IN} {self.h * PT_PER_IN}")
        white = sg.fromstring(
            f'<svg xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="0" y="0" width="{self.w * PT_PER_IN}" '
            f'height="{self.h * PT_PER_IN}" fill="#ffffff"/></svg>').getroot()
        fig.append([white, *self._els])
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.save(str(out))
        return out

    def stack(self, items: list[dict], *, top: float = 0.0, gap: float = 0.10,
              width: float | None = None, letter_gap: float = 0.16,
              letter_size: float = 9.0) -> float:
        """Place panels in a single column, each centred on the same line.

        ``items`` entries take ``svg`` (path), optional ``png``, optional
        ``width``, optional ``letter`` and ``title``. Heights come from the
        measured ink, so the caller never hand-tunes a height ratio, which is
        what previously drifted out of sync with the artwork.

        Returns the y where the stack ends, so a caller can size the canvas to
        its content rather than guessing.
        """
        y = top
        for it in items:
            if it.get("letter"):
                self.add_text(f"{it['letter']}\u2003{it.get('title', '')}".rstrip(),
                              self._letter_x, y + letter_size / PT_PER_IN,
                              size=letter_size)
                y += letter_gap
            p = self.add_svg(it["svg"], y=y, width=it.get("width", width),
                             png=it.get("png"), name=it.get("name"))
            y = p.y + p.h + gap
        return y - gap

    _letter_x = 0.03

    def assert_no_overlap(self, gap_in: float = 0.0) -> None:
        """No two panels may occupy the same vertical space.

        A stacked figure that overflows its canvas does not error, it just
        prints one panel on top of another, which is how a test render came
        back with a head montage sitting across a timeline bar.
        """
        rows = sorted(self.placed, key=lambda p: p.y)
        bad = []
        for a, b in zip(rows, rows[1:]):
            if b.y < a.y + a.h + gap_in:
                bad.append(f"{a.name} ends {a.y + a.h:.3f}in, "
                           f"{b.name} starts {b.y:.3f}in")
        if bad:
            raise AssertionError("panels overlap: " + "; ".join(bad))

    def assert_within_canvas(self) -> None:
        for p in self.placed:
            if p.y + p.h > self.h + 1e-6 or p.x < -1e-6 or p.x + p.w > self.w + 1e-6:
                raise AssertionError(
                    f"{p.name} runs off the canvas: x=[{p.x:.3f},{p.x + p.w:.3f}] "
                    f"y=[{p.y:.3f},{p.y + p.h:.3f}] on {self.w}x{self.h}in")

    def assert_centred(self, names: list[str] | None = None,
                       tol_in: float = 0.01) -> None:
        """Every named panel shares the canvas centre line.

        The defect this catches is cosmetic and silent: nothing errors when a
        panel is 4% off, it just looks wrong on the page, which is how it
        shipped last time.
        """
        want = self.centre_line
        bad = []
        for p in self.placed:
            if names is not None and p.name not in names:
                continue
            if abs(p.cx - want) > tol_in:
                bad.append(f"{p.name} centre {p.cx:.3f}in vs {want:.3f}in")
        if bad:
            raise AssertionError("panels off the centre line: " + "; ".join(bad))
