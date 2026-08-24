"""Composite generated art with CSV-backed data panels at IEEE column width.

Division of labour, and why
---------------------------
FigureLabs draws the schematics: heads, montages, field surfaces, flowcharts.
It is better at those than anything authored here.

matplotlib draws anything carrying a number, because a plot read from CSV can
be audited back to the row it came from and a generated picture cannot. That
distinction already caught one real error: a hand-typed ridge value of 0.517
at horizon 16, whose measured value is 0.822.

What this module guarantees
---------------------------
1. Canvas authored at 3.5 in, the IEEE column measure, so the figure is placed
   at 100% and never rescaled. Rescaling is what produced the blur complaint.
2. Every placed panel prints at >= MIN_DPI. Enforced, not assumed. A panel that
   came back from the generator too small fails the build by name.
3. Panel letters and captions are authored here, not baked into the art, so
   type size is controlled at final scale rather than inherited from whatever
   canvas the generator happened to use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ART = BASE / "art"
VEC = BASE / "art_vector"
TARGET_PX = 2400    # px across the placed panel; 3.5in x 600dpi = 2100, with headroom
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))

COL_IN = 3.5        # IEEE single column
MIN_DPI = 600.0     # print floor for line art
PT_PANEL = 9.0     # panel letters/titles, larger for the vertical stack
PT_NOTE = 8.0
INK = "#1a1a1a"
GREY = "#5f6368"


def _font() -> str:
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


class MissingArt(Exception):
    """Raised when a required generated panel has not been dropped in yet."""


class ResolutionFail(Exception):
    """Raised when a panel is too small to print at MIN_DPI."""


def load_art(name: str):
    """Prefer the traced vector, falling back to the raster original.

    The generator caps output near 1.5 MP, which is under 400 dpi at column
    width. vectorize_art.py traces that flat art to paths, so it can be
    rasterised here at any density. Upscaling the raster was measured and
    rejected: it retained only 82.7% of edge contrast, i.e. visible softening.
    """
    import subprocess
    import tempfile

    import matplotlib.image as mpimg

    svg = VEC / (Path(name).stem + ".svg")
    if svg.exists():
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "r.png"
            # The drawing may occupy only part of its canvas, so sizing the
            # canvas to TARGET_PX can still leave the ink short. Render once,
            # measure the trimmed ink, then re-render scaled up to hit the
            # target on the artwork itself rather than on the whitespace.
            r = subprocess.run(
                ["magick", "-background", "white", str(svg),
                 "-resize", f"{TARGET_PX}x", str(png)], capture_output=True)
            if r.returncode == 0 and png.exists():
                art = _trim(mpimg.imread(png))
                if art.shape[1] < TARGET_PX:
                    grow = min(TARGET_PX / max(art.shape[1], 1) * 1.02, 4.0)
                    r2 = subprocess.run(
                        ["magick", "-background", "white", str(svg),
                         "-resize", f"{int(TARGET_PX * grow)}x", str(png)],
                        capture_output=True)
                    if r2.returncode == 0 and png.exists():
                        art = _trim(mpimg.imread(png))
                return art

    p = ART / name
    if not p.exists():
        raise MissingArt(name)
    return _trim(mpimg.imread(p))


def check_dpi(img, placed_in: float, name: str) -> str | None:
    """Return a failure string if this panel cannot print at MIN_DPI."""
    px = img.shape[1]
    dpi = px / placed_in
    if dpi < MIN_DPI:
        need = int(MIN_DPI * placed_in)
        return (f"{name}: {px}px placed at {placed_in:.2f}in = {dpi:.0f} dpi, "
                f"below {MIN_DPI:.0f}. Regenerate at >= {need}px wide.")
    return None



def _trim(img, thresh: float = 0.985):
    """Crop uniform near-white margin and flatten alpha onto white.

    The generator returns art on its own canvas with generous padding and a
    slightly off-white background. Placed as-is, that padding becomes dead
    space and the off-white reads as a visible grey box against the page.
    Cropping to the ink means the art fills its panel, so it survives
    reduction to column width.
    """
    import numpy as np

    a = np.asarray(img, dtype=float)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    if a.max() > 1.0:
        a = a / 255.0
    if a.shape[2] == 4:                      # composite over white
        alpha = a[..., 3:4]
        a = a[..., :3] * alpha + (1.0 - alpha)
    a = a[..., :3]

    ink = a.min(axis=2) < thresh
    if not ink.any():
        return a
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    pad = 2
    r0 = max(rows[0] - pad, 0); r1 = min(rows[-1] + 1 + pad, a.shape[0])
    c0 = max(cols[0] - pad, 0); c1 = min(cols[-1] + 1 + pad, a.shape[1])
    out = a[r0:r1, c0:c1].copy()
    # Measured background is 254/255, which reproduces as a faint grey panel
    # against the page. Snap that near-white plateau to true white and leave
    # every real ink value untouched.
    bg = out.min(axis=2) >= 0.975
    out[bg] = 1.0
    return out


# Left margin of the figure, in figure fraction, and the gutter the data panel
# needs for its y tick labels. Both are reserved up front.
#
# Two alignment bugs were measured here, not guessed at:
#
#   1. savefig(bbox_inches="tight") grows the canvas leftward to fit the data
#      panel's y tick labels. The art panels are centred on the authored
#      canvas, so once the canvas grows on one side only, their ink lands right
#      of the figure's visual centre. Measured: Fig 3 art sat at 57.1% of
#      width against the bar chart at 50.0%, which is the shift that reads as
#      "the top image is a bit to the right". Fixed by reserving the gutter in
#      the layout and saving with a fixed bbox, so the canvas never grows.
#
#   2. imshow forces an equal aspect, which shrinks the axes box to the image
#      and re-centres it. A panel letter anchored to that box therefore indents
#      by a different amount on every row. Fixed by anchoring letters to a
#      single figure-space x, so A, B and C share one left edge.
LEFT = 0.030
RIGHT = 0.985
# Per-figure, because the widest y tick label differs: Fig 1's sweep carries
# numeric labels, Fig 3's bar chart carries category names ("mean of trace").
# Measured by rendering and comparing each data axes' tight bbox against its
# position, then rounded up. build() re-checks and fails loudly if a label
# grows past what is reserved.
GUTTER_SWEEP = 0.125
GUTTER_BARS = 0.225


def _letter(fig, ax, letter: str, title: str) -> None:
    """Panel letter at the figure's left margin, not the axes' left edge."""
    y = ax.get_position().y1
    fig.text(LEFT, y + 0.006, f"{letter}\u2003{title}",
             fontsize=PT_PANEL, fontweight="bold", color=INK,
             ha="left", va="bottom", gid="chrome")


def place(fig, ax, img, letter: str, title: str) -> None:
    ax.imshow(img)
    ax.axis("off")
    _letter(fig, ax, letter, title)


def build_figure1():
    """Fig 1: generated montage + generated field, over the measured sweep."""
    import make_data_panels as dp

    head = load_art("head_montage.png")
    field = load_art("latent_field.png")

    fails = [f for f in (
        check_dpi(head, COL_IN, "head_montage.png"),
        check_dpi(field, COL_IN, "latent_field.png"),
    ) if f]
    if fails:
        raise ResolutionFail("; ".join(fails))

    plt.rcParams.update({"font.family": _font(), "pdf.fonttype": 42})
    # One panel per row: each gets the full column measure, so the art is
    # placed at 3.5in rather than 1.75in and its type prints twice as large.
    fig = plt.figure(figsize=(COL_IN, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.30, 0.92, 1.15],
                          left=LEFT + GUTTER_SWEEP, right=RIGHT,
                          top=0.955, bottom=0.062, hspace=0.22)

    place(fig, fig.add_subplot(gs[0, 0]), head, "A", "Noninvasive EEG")
    place(fig, fig.add_subplot(gs[1, 0]), field, "B", "Latent field")

    ax = fig.add_subplot(gs[2, 0])
    dp.panel_sweep(ax, dp.load())
    _letter(fig, ax, "C", "Measured forecast skill")

    return fig


def build_figure3():
    """Fig 3: generated protocol schematics over the measured comparison."""
    import make_data_panels as dp

    overlap = load_art("overlap.png")
    split = load_art("subject_split.png")

    fails = [f for f in (
        check_dpi(overlap, COL_IN, "overlap.png"),
        check_dpi(split, COL_IN, "subject_split.png"),
    ) if f]
    if fails:
        raise ResolutionFail("; ".join(fails))

    plt.rcParams.update({"font.family": _font(), "pdf.fonttype": 42})
    fig = plt.figure(figsize=(COL_IN, 5.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.00, 0.72, 1.18],
                          left=LEFT + GUTTER_BARS, right=RIGHT,
                          top=0.950, bottom=0.072, hspace=0.26)

    place(fig, fig.add_subplot(gs[0, 0]), overlap, "A", "Leakage risk")
    place(fig, fig.add_subplot(gs[1, 0]), split, "B", "Held-out split")

    ax = fig.add_subplot(gs[2, 0])
    dp.panel_bars(ax, dp.load(), h=16)
    _letter(fig, ax, "C", "Strong-baseline comparison")

    return fig


def build_figure2():
    """Figure 2 is a single generated flowchart, placed full column."""
    art = load_art("architecture.png")
    fail = check_dpi(art, COL_IN, "architecture.png")
    if fail:
        raise ResolutionFail(fail)

    h = COL_IN * art.shape[0] / art.shape[1]
    fig = plt.figure(figsize=(COL_IN, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(art)
    ax.axis("off")
    return fig


def assert_gutter(fig, tag: str) -> None:
    """The reserved gutter must contain the tick labels.

    Without this the failure is silent and cosmetic: matplotlib clips nothing,
    it just draws the label off-canvas, and the figure looks fine in a
    thumbnail while a label is missing in print.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[-1]
    tb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    if tb.x0 < 0:
        raise ResolutionFail(
            f"{tag}: y tick labels overflow the canvas by "
            f"{-tb.x0:.3f} of figure width; raise the gutter")


def assert_aligned(fig, tag: str, tol_px: float = 2.0) -> None:
    """Every panel's content must share one vertical centre line.

    Compares the drawn image rect of each art panel against the data panel's
    spine box. This is the check for the defect being fixed: art centred on
    its own canvas while the chart is centred on the column reads as one
    panel shoved sideways.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    centres = []
    for ax in fig.axes:
        if ax.get_aspect() == "auto":
            b = ax.get_window_extent()
        else:
            b = ax.get_images()[0].get_window_extent(r)
        centres.append((b.x0 + b.x1) / 2)
    spread = max(centres) - min(centres)
    if spread > tol_px:
        raise ResolutionFail(
            f"{tag}: panel centres differ by {spread:.1f}px "
            f"({[round(c, 1) for c in centres]})")


def main() -> int:
    jobs = [("fig1_concept", build_figure1),
            ("fig2_architecture", build_figure2),
            ("fig3_protocol", build_figure3)]

    done, waiting = [], []
    for name, fn in jobs:
        try:
            fig = fn()
        except MissingArt as e:
            waiting.append(f"{name}: waiting on art/{e}")
            continue
        except ResolutionFail as e:
            waiting.append(f"{name}: {e}")
            continue
        assert_gutter(fig, name)
        assert_aligned(fig, name)
        for ext in ("pdf", "png"):
            # No bbox_inches="tight": a tight bbox grows the canvas leftward
            # to fit y tick labels, which is what pushed the art panels right
            # of the data panel. The layout already reserves that space.
            fig.savefig(OUT / f"{name}.{ext}",
                        dpi=600 if ext == "png" else None)
        plt.close(fig)
        done.append(name)

    for n in done:
        print(f"  built {n}.pdf / .png")
    for w in waiting:
        print(f"  {w}")

    if done:
        print("\nRun check_geometry.py to gate type size and overlap.")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
