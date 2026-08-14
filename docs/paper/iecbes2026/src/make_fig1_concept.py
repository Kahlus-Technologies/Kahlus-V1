"""Figure 1: Kahlus-V1 concept, redrawn for IEEE single-column placement.

WHY THIS EXISTS
---------------
The version in "Paper - Aayushya - Krish - Zaidi - July 27-2026 Version 6" is a
raster crop of a wide 1376x768 slide. Placed at IEEE column width (3.5 in) it
prints at roughly 390 dpi of *source* pixels but with glyphs that were drawn for
a 13-inch slide, so the type lands near 5 pt and reads as blurred. Dr. Zaidi's
note on the 2026-08-03 call was exactly this: "I'm wondering why it's so low
quality... it's very blurred", while holding the figure selected at ~2 in wide.

Cropping cannot fix it. The existing crop-only pipeline
(docs/paper/vertical_figures/src/compose.py) says so in its own docstring:
printed_pt = glyph_px / canvas_px * 72 * placed_width_in, which has no term for
rows, so restacking never enlarges type. The fix is to re-render the artwork
with type sized for the final placement. That is what this module does.

DESIGN CONTRACT
---------------
- Authored at FIG_W_IN wide, the true IEEE single-column width, so nothing is
  rescaled downstream. Place at 100%.
- Every text element is >= 7 pt at final size. IEEE's floor for figure text is
  effectively 6 pt; 7 pt leaves margin for the printer.
- Vector PDF is the deliverable. The PNG/TIFF exports are 600 dpi for editors
  who need raster.
- 2x2 panel grid (A/B/C/D) rather than 4-across, so the figure occupies one
  column instead of spanning both.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt  # re-exported for the geometry gate
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# IEEE two-column letter: 8.5in - 2x0.625in margin - 0.25in gutter => 3.5in.
FIG_W_IN = 3.5
FIG_H_IN = 3.45

# Type scale. These are final printed sizes because the canvas is authored at
# the placement width, so no scaling intervenes.
PT_PANEL = 8.0   # panel letter + title
PT_BODY = 7.0    # in-box label
PT_NOTE = 7.0    # caption-ish annotation under a panel

INK = "#1a1a1a"
TEAL = "#0b6b63"
BLUE = "#1f4e9c"
GREEN = "#1b7a3d"
RED = "#b3261e"
GREY = "#5f6368"
FRAME = "#c8ccd0"


def _font() -> str:
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def _mark(ax, x, y, ok: bool, color, fs):
    """Draw a check/ballot-x from DejaVu Sans.

    Arial lacks U+2713 and U+2717, so matplotlib silently renders a .notdef box.
    Only the mark is pinned to DejaVu; body text stays in the document face.
    """
    ax.text(x, y, "\u2713" if ok else "\u2717", fontsize=fs, color=color,
            fontweight="bold", ha="center", va="center",
            fontfamily="DejaVu Sans", transform=ax.transAxes)


def _panel(ax, letter: str, title: str) -> None:
    """Draw the panel letter and title in the reserved header band."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.035, 0.955, letter,
        fontsize=PT_PANEL, fontweight="bold", color=INK,
        ha="left", va="top", gid="chrome",
    )
    ax.text(
        0.115, 0.955, title,
        fontsize=PT_PANEL, fontweight="bold", color=INK,
        ha="left", va="top", gid="chrome",
    )


def _box(ax, x, y, w, h, text, *, ec, fc="white", fs=PT_BODY, weight="normal",
         color=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=0.8, edgecolor=ec, facecolor=fc,
            transform=ax.transAxes, clip_on=False,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        fontsize=fs, color=color, fontweight=weight,
        ha="center", va="center", linespacing=1.25,
        transform=ax.transAxes,
    )


def _arrow(ax, x0, y0, x1, y1, color=INK):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=7,
            linewidth=0.9, color=color,
            transform=ax.transAxes, clip_on=False,
        )
    )


def panel_a(ax) -> None:
    """Noninvasive EEG acquisition: the observed side of the model.

    Layout note: the scalp sits in the upper half and the acquired-window box
    directly beneath it, so the arrow reads top-to-bottom like the rest of the
    figure. An earlier side-by-side arrangement collided with the montage.
    """
    _panel(ax, "A", "Noninvasive EEG")

    cx, cy, r = 0.50, 0.665, 0.185

    ax.add_patch(mpatches.Circle((cx, cy), r, transform=ax.transAxes,
                                 fill=False, edgecolor=GREY, linewidth=0.9))
    # Nasion bump, so orientation is unambiguous.
    ax.add_patch(mpatches.Arc((cx, cy + r), 0.075, 0.055, theta1=0, theta2=180,
                              transform=ax.transAxes, edgecolor=GREY,
                              linewidth=0.9))

    # 22 electrodes, inset inside the scalp boundary. The inset factor keeps
    # every marker strictly within the outline: at row height dy the chord
    # half-width is r*sqrt(1-(dy/r)^2), and 0.78 leaves the marker clear of it.
    rows = [(0.78, 3), (0.42, 5), (0.0, 6), (-0.42, 5), (-0.80, 3)]
    for frac, n in rows:
        dy = frac * r
        half = 0.78 * r * np.sqrt(max(0.0, 1.0 - frac ** 2))
        xs = np.linspace(cx - half, cx + half, n) if n > 1 else [cx]
        for x in xs:
            ax.add_patch(mpatches.Circle((x, cy + dy), 0.0155,
                                         transform=ax.transAxes,
                                         facecolor="white", edgecolor=TEAL,
                                         linewidth=0.7))

    _arrow(ax, cx, 0.442, cx, 0.375, color=TEAL)
    _box(ax, 0.11, 0.185, 0.78, 0.185,
         "Past window: 22 ch x 128", ec=TEAL, fc="#e8f3f1", weight="bold",
         color=TEAL)

    ax.text(0.5, 0.075, "high-dimensional, noisy,\nsubject-specific",
            fontsize=PT_NOTE, color=GREY, style="italic",
            ha="center", va="center", linespacing=1.3,
            transform=ax.transAxes)


def panel_b(ax) -> None:
    """Latent neural field: the hidden state the model actually evolves."""
    _panel(ax, "B", "Latent neural field")

    # A smooth 2-D field surface, drawn as a wireframe so it stays legible in
    # grayscale print.
    gx = np.linspace(-2.2, 2.2, 26)
    gy = np.linspace(-2.2, 2.2, 26)
    X, Y = np.meshgrid(gx, gy)
    Z = (np.exp(-((X + 0.75) ** 2 + (Y + 0.35) ** 2) / 1.5)
         + 0.75 * np.exp(-((X - 0.95) ** 2 + (Y - 0.75) ** 2) / 1.1)
         - 0.35 * np.exp(-((X - 0.35) ** 2 + (Y + 1.15) ** 2) / 0.9))

    # Oblique projection into axes coordinates.
    px = 0.50 + 0.105 * X + 0.036 * Y
    py = 0.50 + 0.052 * Y + 0.150 * Z
    for i in range(0, X.shape[0], 2):
        ax.plot(px[i], py[i], color=BLUE, linewidth=0.35, alpha=0.75,
                transform=ax.transAxes, clip_on=False)
    for j in range(0, X.shape[1], 2):
        ax.plot(px[:, j], py[:, j], color=BLUE, linewidth=0.35, alpha=0.75,
                transform=ax.transAxes, clip_on=False)

    ax.text(0.5, 0.815, r"hidden brain state $F_s$",
            fontsize=PT_BODY, color=BLUE, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.105, "inferred, not measured;\nevolved causally",
            fontsize=PT_NOTE, color=GREY, style="italic",
            ha="center", va="center", linespacing=1.3, transform=ax.transAxes)


def panel_c(ax) -> None:
    """Readout to a strictly future window. Disjointness is the whole point."""
    _panel(ax, "C", "Future-window readout")

    _box(ax, 0.05, 0.575, 0.90, 0.185,
         "Field readout operator", ec=BLUE, fc="#eaf0fa", color=BLUE)
    _arrow(ax, 0.5, 0.555, 0.5, 0.485, color=BLUE)
    _box(ax, 0.05, 0.265, 0.90, 0.20,
         "Future window\ndisjoint from input", ec=GREEN, fc="#e9f5ec",
         weight="bold", color=GREEN)

    ax.text(0.5, 0.105,
            "input and target index sets\nasserted disjoint",
            fontsize=PT_NOTE, color=GREY, style="italic",
            ha="center", va="center", linespacing=1.3, transform=ax.transAxes)


def panel_d(ax) -> None:
    """Evidence gate. Reports what the paper is allowed to conclude.

    Deliberately does NOT show MSE=3.116 / r=0.972. Section VI of the paper
    retracts those as an artifact of window overlap, so putting them in a
    figure captioned "APPROVED" would contradict the manuscript.
    """
    _panel(ax, "D", "Evidence gate")

    for y, ok, label, color, fill in (
        (0.585, True, "beats persistence,\nmean-of-trace, random", GREEN, "#e9f5ec"),
        (0.315, False, "superiority over\nlinear ridge AR", RED, "#fdeceb"),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (0.04, y), 0.92, 0.22,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                linewidth=0.8, edgecolor=color, facecolor=fill,
                transform=ax.transAxes, clip_on=False,
            )
        )
        _mark(ax, 0.125, y + 0.11, ok, color, PT_BODY + 1)
        ax.text(0.215, y + 0.11, label, fontsize=PT_BODY, color=color,
                fontweight="bold", ha="left", va="center", linespacing=1.25,
                transform=ax.transAxes)

    ax.text(0.5, 0.135, "gate blocked an overlap-inflated\nresult from this pipeline",
            fontsize=PT_NOTE, color=GREY, style="italic",
            ha="center", va="center", linespacing=1.3, transform=ax.transAxes)


def build_figure():
    """Compose and return the live figure, for inspection by check_geometry."""
    plt.rcParams.update({
        "font.family": _font(),
        "pdf.fonttype": 42,   # embed TrueType, keep text selectable
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.012,
    })

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = fig.add_gridspec(
        2, 2,
        left=0.015, right=0.985, top=0.985, bottom=0.015,
        wspace=0.10, hspace=0.14,
    )

    for idx, fn in enumerate((panel_a, panel_b, panel_c, panel_d)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        fn(ax)
        ax.add_patch(
            FancyBboxPatch(
                (0.0, 0.0), 1.0, 1.0,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                linewidth=0.7, edgecolor=FRAME, facecolor="none",
                transform=ax.transAxes, clip_on=False, zorder=-5,
            )
        )

    return fig


def build() -> Path:
    fig = build_figure()
    pdf = OUT / "fig1_concept.pdf"
    fig.savefig(pdf)
    fig.savefig(OUT / "fig1_concept.png", dpi=600)
    fig.savefig(OUT / "fig1_concept.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return pdf


if __name__ == "__main__":
    print(build())
