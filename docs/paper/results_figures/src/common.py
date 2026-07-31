"""Shared plotting setup for the IEEE column-width results figures.

The whole point of this module is the width. The paper's earlier figures were
authored at figsize=(16.5, 4.5) with fontsize=8 and then shrunk into a 3.25 in
column, which divides every glyph by 5.08 and prints 8 pt type at 1.58 pt. No
amount of cropping or re-stacking fixes that, because printed size is

    printed_pt = glyph_px / canvas_px * 72 * placed_in

and the row count never enters. The only durable fix is to author at the width
the figure is actually placed at, which is what COL_W does. Every figure here
sets its font sizes in points and is placed at its authored width, so what
matplotlib calls 8 pt is 8 pt on paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# IEEE two-column: 3.25 in usable column width.
COL_W = 3.25

ROOT = Path(__file__).resolve().parents[4]
ART = ROOT / "artifacts"
OUT = Path(__file__).resolve().parents[1] / "figures"

# Point sizes. These are real points because the figure is authored at the
# width it is printed at; see the module docstring.
PT_TICK = 7.0
PT_LABEL = 8.0
PT_TITLE = 8.5
PT_LEGEND = 7.0
PT_ANNOT = 6.5

# Colour-blind safe, and distinguishable in greyscale by lightness order.
C_MODEL = "#1b6ca8"     # the learned model
C_RIDGE = "#e07b39"     # tuned linear baseline
C_PERSIST = "#7a7a7a"   # persistence
C_MEAN = "#b8b8b8"      # mean-of-trace
C_BAD = "#b3272d"
C_GOOD = "#2e7d32"


def rc() -> None:
    """Apply the shared rcParams. Called once per figure module."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": PT_LABEL,
        "axes.labelsize": PT_LABEL,
        "axes.titlesize": PT_TITLE,
        "xtick.labelsize": PT_TICK,
        "ytick.labelsize": PT_TICK,
        "legend.fontsize": PT_LEGEND,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        # NOT bbox="tight". Tight bbox grows the canvas past figsize to fit
        # labels, so a figure authored at 3.25 in saves at 4.08 in, and placing
        # it in a 3.25 in column scales it by 0.80 and prints 8 pt type at
        # 6.4 pt. That is the same defect as the old 16.5 in figures, just
        # smaller. Constrained layout fits the labels by shrinking the axes
        # inside a fixed canvas instead, so the saved width is exactly figsize.
        "savefig.bbox": None,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    })


def load(rel: str) -> dict:
    """Read an artifact JSON by path relative to artifacts/."""
    with (ART / rel).open() as f:
        return json.load(f)


def save(fig, name: str, manifest: dict) -> Path:
    """Write the figure plus the data manifest that the verifier checks.

    The manifest exists so that every plotted number can be traced back to the
    artifact it came from. verify_figures.py re-reads the artifact and compares
    elementwise, which is what stops a hand-typed number from surviving an edit.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path)
    fig.savefig(OUT / f"{name}.pdf")

    # The whole point of this pipeline is that the saved width equals the
    # placement width, so that authored point sizes are printed point sizes.
    # Assert it here rather than trusting it, because bbox settings and layout
    # engines have both silently broken it before.
    w_in = fig.get_size_inches()[0]
    if abs(w_in - COL_W) > 1e-6:
        raise AssertionError(
            f"{name}: saved width {w_in:.3f} in != column width {COL_W} in; "
            "placing this would rescale the type"
        )

    manifest = dict(manifest)
    manifest["figure_width_in"] = float(w_in)
    manifest["figure_height_in"] = float(fig.get_size_inches()[1])
    manifest["min_authored_pt"] = float(min(PT_ANNOT, PT_TICK))
    plt.close(fig)
    with (OUT / f"{name}.json").open("w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return path
