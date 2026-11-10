"""Figure 3: evaluation-protocol figure, re-rendered as a 2x2 IEEE column block.

WHY THIS EXISTS
---------------
The Version 6 manuscript carries this figure as word/media/image4.png, a
1339x333 strip of four panels laid out 4-across. At that aspect ratio it can
only be placed as a double-column float, and its type prints near 4 pt.

On the 2026-08-03 call Dr. Zaidi asked, with the cursor tracing a vertical line
down the middle of the strip: "cut it into half from here... half and then half
under it. So that will give us ABCD. So they can appear in one column."

This module does that, but by re-rendering rather than slicing, because slicing
a 333px-tall strip into quadrants leaves 4 pt type at 4 pt. The panel order and
content match the original: A leakage risk, B subject-held-out split, C strong
baseline comparison, D claim gate.

NUMBERS SHOWN HERE
------------------
Panel C plots the isolated, non-overlapping, subject-held-out results, which are
the numbers Section VI of the paper actually stands behind:
    h=16: recurrent MSE 0.690 vs persistence 1.334
    h=1 : ridge 0.517 vs recurrent 0.568, bootstrap CI [-0.086, -0.016]
It deliberately does not plot the overlapping-window MSE=3.116 / r=0.972 result,
which the manuscript retracts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
from matplotlib import pyplot as plt  # re-exported for the geometry gate
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FIG_W_IN = 3.5
FIG_H_IN = 3.30

PT_PANEL = 8.0
PT_BODY = 7.0
PT_TICK = 7.0
PT_NOTE = 7.0

INK = "#1a1a1a"
GREY = "#5f6368"
RED = "#b3261e"
GREEN = "#1b7a3d"
BLUE = "#1f4e9c"
AMBER = "#a86a00"
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
    ax.text(0.0, 1.075, letter, fontsize=PT_PANEL, fontweight="bold",
            color=INK, ha="left", va="bottom", transform=ax.transAxes,
            gid="chrome")
    ax.text(0.085, 1.075, title, fontsize=PT_PANEL, fontweight="bold",
            color=INK, ha="left", va="bottom", transform=ax.transAxes,
            gid="chrome")


def panel_a(ax) -> None:
    """Window overlap is the failure mode the gate caught."""
    _panel(ax, "A", "Leakage risk")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Overlapping case: input and target share W-H samples.
    ax.add_patch(Rectangle((0.06, 0.575), 0.52, 0.15, facecolor="#dce6f7",
                           edgecolor=BLUE, linewidth=0.8))
    ax.add_patch(Rectangle((0.42, 0.575), 0.52, 0.15, facecolor="#f7dcda",
                           edgecolor=RED, linewidth=0.8, alpha=0.85))
    ax.add_patch(Rectangle((0.42, 0.575), 0.16, 0.15, facecolor=RED,
                           edgecolor="none", alpha=0.30))
    ax.text(0.50, 0.790, "overlapping", fontsize=PT_BODY, color=RED,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.50, 0.545, "shared samples \u2192 inflated r", fontsize=PT_NOTE,
            color=RED, ha="center", va="top")

    # Disjoint case: what this paper scores.
    ax.add_patch(Rectangle((0.06, 0.205), 0.42, 0.145, facecolor="#dce6f7",
                           edgecolor=BLUE, linewidth=0.8))
    ax.add_patch(Rectangle((0.52, 0.205), 0.42, 0.145, facecolor="#dcefe1",
                           edgecolor=GREEN, linewidth=0.8))
    ax.text(0.27, 0.2775, "input", fontsize=PT_TICK, color=BLUE,
            ha="center", va="center")
    ax.text(0.73, 0.2775, "target", fontsize=PT_TICK, color=GREEN,
            ha="center", va="center")
    ax.text(0.50, 0.375, "disjoint (scored)", fontsize=PT_BODY, color=GREEN,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.50, 0.135, "strictly future sample only", fontsize=PT_NOTE,
            color=GREY, style="italic", ha="center", va="top", gid="chrome")


def panel_b(ax) -> None:
    """Subjects, not windows, are the unit of held-out evaluation."""
    _panel(ax, "B", "Subject-held-out")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i in range(7):
        x = 0.075 + i * 0.108
        held = i >= 5
        ax.add_patch(Rectangle((x, 0.545), 0.082, 0.185,
                               facecolor="#f6e2c8" if held else "#dce6f7",
                               edgecolor=AMBER if held else BLUE,
                               linewidth=0.8))
        ax.text(x + 0.041, 0.6375, f"S{i + 1}", fontsize=PT_TICK,
                color=AMBER if held else BLUE, ha="center", va="center")

    ax.text(0.28, 0.775, "train", fontsize=PT_BODY, color=BLUE,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.83, 0.775, "held out", fontsize=PT_BODY, color=AMBER,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.5, 0.395, "no subject appears\non both sides",
            fontsize=PT_NOTE, color=GREY, ha="center", va="center",
            linespacing=1.3)
    ax.text(0.5, 0.170, "n = 2 test subjects,\nbelow powered threshold",
            fontsize=PT_NOTE, color=RED, style="italic",
            ha="center", va="center", linespacing=1.3)


def panel_c(ax) -> None:
    """The comparison that decides the paper's claim."""
    _panel(ax, "C", "Strong baselines")

    labels = ["persist.", "ridge", "ours"]
    vals = [1.334, 0.517, 0.690]
    colors = [GREY, GREEN, BLUE]

    bars = ax.bar(range(3), vals, color=colors, width=0.62,
                  edgecolor="none")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.055, f"{v:.3f}",
                fontsize=PT_TICK, color=INK, ha="center", va="bottom")

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=PT_TICK)
    ax.set_ylabel("MSE", fontsize=PT_BODY, labelpad=1.5)
    ax.set_ylim(0, 1.62)
    ax.tick_params(axis="y", labelsize=PT_TICK, length=2, pad=1)
    ax.tick_params(axis="x", length=0, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(GREY)

    ax.text(0.5, -0.205, "isolated, subject-held-out",
            fontsize=PT_NOTE, color=GREY, style="italic",
            ha="center", va="top", transform=ax.transAxes, gid="chrome")


def panel_d(ax) -> None:
    """What the gate lets the paper say, and what it does not."""
    _panel(ax, "D", "Claim gate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rows = [
        (0.72, True, "beats persistence", GREEN, "#e9f5ec"),
        (0.47, True, "protocol caught leak", GREEN, "#e9f5ec"),
        (0.22, False, "beats linear ridge", RED, "#fdeceb"),
    ]
    for y, mark, text, color, fill in rows:
        ax.add_patch(
            FancyBboxPatch(
                (0.04, y), 0.92, 0.19,
                boxstyle="round,pad=0.006,rounding_size=0.03",
                linewidth=0.8, edgecolor=color, facecolor=fill,
                transform=ax.transAxes, clip_on=False,
            )
        )
        _mark(ax, 0.115, y + 0.095, mark, color, PT_BODY + 0.5)
        ax.text(0.205, y + 0.095, text, fontsize=PT_BODY, color=INK,
                ha="left", va="center")

    ax.text(0.5, 0.045, "claims released only\nwhen gate passes",
            fontsize=PT_NOTE, color=GREY, style="italic",
            ha="center", va="center", linespacing=1.3, gid="chrome")


def build_figure():
    """Compose and return the live figure, for inspection by check_geometry."""
    plt.rcParams.update({
        "font.family": _font(),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.012,
    })

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    gs = fig.add_gridspec(
        2, 2,
        left=0.115, right=0.965, top=0.945, bottom=0.105,
        wspace=0.40, hspace=0.30,
    )

    for idx, fn in enumerate((panel_a, panel_b, panel_c, panel_d)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        fn(ax)

    return fig


def build() -> Path:
    fig = build_figure()
    pdf = OUT / "fig3_protocol.pdf"
    fig.savefig(pdf)
    fig.savefig(OUT / "fig3_protocol.png", dpi=600)
    fig.savefig(OUT / "fig3_protocol.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return pdf


if __name__ == "__main__":
    print(build())
