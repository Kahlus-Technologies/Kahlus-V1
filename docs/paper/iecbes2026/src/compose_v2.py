"""Compose the paper's figures on an absolute-placement canvas.

Layout is declarative here: each figure is a stack of panels, every panel is
placed by its measured ink, and the whole stack shares one centre line. The
previous version expressed the same intent through matplotlib gridspec and
hand-tuned height ratios, which produced two silent misalignments (see
canvas.py for the measurements).

Everything stays vector from art to PDF. The PNG is a rendering of the same
SVG, so what is checked is what ships.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ART = BASE / "art"
VEC = BASE / "art_vector"
OUT = BASE / "figures"
TMP = BASE / ".build"

sys.path.insert(0, str(HERE))

from canvas import Canvas          # noqa: E402
from render import svg_to_pdf, svg_to_png   # noqa: E402

COL_IN = 3.5        # IEEE single column; the document measure is applied later
GAP = 0.10
LETTER_GAP = 0.17
PT_LETTER = 9.0


def data_panel_svg(which: str, width_in: float, height_in: float) -> Path:
    """Render one CSV-backed matplotlib panel to SVG.

    Data panels stay in matplotlib because a plot that can be audited back to
    the row it came from is worth more than one that is drawn by hand. Only
    the *layout* moved off matplotlib.
    """
    import make_data_panels as dp

    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"{which}.svg"
    plt.rcParams.update({"svg.fonttype": "none"})
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0.30, 0.16, 0.68, 0.80])
    data = dp.load()
    if which == "sweep":
        dp.panel_sweep(ax, data)
    else:
        dp.panel_bars(ax, data, h=16)
    fig.savefig(out, format="svg", transparent=True)
    plt.close(fig)
    return out


def figure1() -> Canvas:
    """Concept: montage, latent field, then the measured sweep."""
    c = Canvas(COL_IN, 10.0)          # trimmed to content below
    sweep = data_panel_svg("sweep", COL_IN, 2.30)
    end = c.stack([
        dict(svg=VEC / "head_montage.svg", png=ART / "head_montage.png",
             letter="A", title="Noninvasive EEG", width=2.55),
        dict(svg=VEC / "latent_field.svg", png=ART / "latent_field.png",
             letter="B", title="Latent field", width=3.05),
        dict(svg=sweep, png=None, letter="C",
             title="Measured forecast skill", width=3.35),
    ], top=0.03, gap=GAP, letter_gap=LETTER_GAP, letter_size=PT_LETTER)
    c.h = end + 0.03
    return c


def figure2() -> Canvas:
    """Architecture: one generated flowchart, full measure."""
    c = Canvas(COL_IN, 10.0)
    end = c.stack([
        dict(svg=VEC / "architecture.svg", png=ART / "architecture.png",
             width=3.35),
    ], top=0.03, gap=GAP)
    c.h = end + 0.03
    return c


def figure3() -> Canvas:
    """Protocol: leakage, split, then the measured comparison."""
    c = Canvas(COL_IN, 10.0)
    bars = data_panel_svg("bars", COL_IN, 2.20)
    end = c.stack([
        dict(svg=VEC / "overlap.svg", png=ART / "overlap.png",
             letter="A", title="Leakage risk", width=3.30),
        dict(svg=VEC / "subject_split.svg", png=ART / "subject_split.png",
             letter="B", title="Held-out split", width=3.20),
        dict(svg=bars, png=None, letter="C",
             title="Strong-baseline comparison", width=3.35),
    ], top=0.03, gap=GAP, letter_gap=LETTER_GAP, letter_size=PT_LETTER)
    c.h = end + 0.03
    return c


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("fig1_concept", figure1),
            ("fig2_architecture", figure2),
            ("fig3_protocol", figure3)]

    for name, fn in jobs:
        c = fn()
        # Gates first: a figure that fails these is not written, so a stale
        # good file is never silently replaced by a broken one.
        c.assert_within_canvas()
        c.assert_no_overlap()
        c.assert_centred()
        svg = c.save_svg(OUT / f"{name}.svg")
        svg_to_pdf(svg, OUT / f"{name}.pdf")
        svg_to_png(svg, OUT / f"{name}.png", dpi=600)
        sizes = ", ".join(f"{p.name} {p.w:.2f}x{p.h:.2f}in" for p in c.placed)
        print(f"  built {name}  {c.w:.2f}x{c.h:.2f}in  [{sizes}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
