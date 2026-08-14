"""Data panels for the IECBES figures, plotted from the measured sweep.

Every number here is read from
    docs/paper/ieee_figure_source/data/amrith_horizon_sweep.csv
at plot time. Nothing is typed in by hand, so a value cannot drift from the
artifact it came from. That property is the reason these panels stay in
matplotlib while the illustrative panels are generated art: a plot can be
audited back to a CSV row, a picture cannot.

CORRECTION LOGGED HERE
----------------------
An earlier revision of fig3 panel C plotted ridge = 0.517 at horizon 16. That
was wrong twice over. The measured ridge value at h=16 is 0.822; 0.517 is the
h=1 single-channel BNCI comparison quoted in the limitations paragraph, a
different experiment entirely. Mixing them understated the model: at h=16 the
recurrent model leads ridge (0.690 vs 0.822), which the erroneous chart
inverted. The loader below refuses to plot a horizon that is not present in the
CSV, so that class of mistake cannot recur silently.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent          # repo root
CSV = ROOT / "docs/paper/ieee_figure_source/data/amrith_horizon_sweep.csv"
OUT = HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PT_BODY = 7.0
PT_TICK = 7.0

INK = "#1a1a1a"
GREY = "#5f6368"
BLUE = "#1f4e9c"
GREEN = "#15803d"
TEAL = "#0f766e"
AMBER = "#b45309"
RED = "#b3261e"

# Display order is worst-to-best so the reader's eye lands on the model last.
SERIES = [
    ("random", "random", RED),
    ("persistence", "persistence", GREY),
    ("mean_of_trace", "mean of trace", AMBER),
    ("ridge_ar", "ridge AR", GREEN),
    ("kahlus_gru", "Kahlus-V1", BLUE),
]


def _font() -> str:
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def load() -> dict[str, dict[int, float]]:
    """Read the sweep into {method: {horizon: mse}}.

    Raises if the file is missing rather than falling back to defaults; a
    silently empty plot is worse than a failed build.
    """
    if not CSV.exists():
        raise FileNotFoundError(f"measured sweep not found: {CSV}")

    out: dict[str, dict[int, float]] = {}
    with CSV.open() as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["method"], {})[int(row["horizon"])] = float(row["mse"])
    if not out:
        raise ValueError(f"no rows parsed from {CSV}")
    return out


def at_horizon(data: dict[str, dict[int, float]], h: int) -> dict[str, float]:
    """Values for one horizon, insisting every series actually has that point."""
    missing = [m for m, _, _ in SERIES if h not in data.get(m, {})]
    if missing:
        raise KeyError(f"horizon {h} absent for: {', '.join(missing)}")
    return {m: data[m][h] for m, _, _ in SERIES}


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(GREY)
    ax.tick_params(labelsize=PT_TICK, length=2, pad=1.5, colors=INK)


def panel_sweep(ax, data) -> None:
    """MSE against forecast horizon, all five methods."""
    for key, label, color in SERIES:
        pts = sorted(data[key].items())
        xs = [h for h, _ in pts]
        ys = [v for _, v in pts]
        lead = key == "kahlus_gru"
        ax.plot(xs, ys,
                marker="o", markersize=2.6 if lead else 2.0,
                linewidth=1.5 if lead else 0.9,
                color=color, label=label,
                zorder=3 if lead else 2)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("forecast horizon (samples)", fontsize=PT_BODY, labelpad=2)
    ax.set_ylabel("MSE", fontsize=PT_BODY, labelpad=2)
    _style(ax)
    ax.legend(fontsize=PT_TICK - 0.5, frameon=False, loc="upper left",
              handlelength=1.4, borderaxespad=0.2, labelspacing=0.25)


def panel_bars(ax, data, h: int = 16) -> None:
    """Single-horizon comparison. Values come from the same CSV as the sweep."""
    vals = at_horizon(data, h)
    order = [(label, vals[key], color) for key, label, color in SERIES]
    order.sort(key=lambda t: -t[1])          # worst first, model last

    labels = [o[0] for o in order]
    ys = [o[1] for o in order]
    colors = [o[2] for o in order]

    bars = ax.barh(range(len(ys)), ys, color=colors, height=0.62)
    for b, v in zip(bars, ys):
        ax.text(v + max(ys) * 0.02, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", fontsize=PT_TICK, color=INK,
                ha="left", va="center")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=PT_TICK)
    ax.set_xlim(0, max(ys) * 1.22)
    ax.set_xlabel(f"MSE at horizon {h}", fontsize=PT_BODY, labelpad=2)
    ax.tick_params(axis="y", length=0)
    _style(ax)


def build() -> list[Path]:
    plt.rcParams.update({
        "font.family": _font(),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.012,
    })
    data = load()
    written: list[Path] = []

    for name, fn, size in (
        ("data_sweep", panel_sweep, (3.5, 2.35)),
        ("data_bars", panel_bars, (3.5, 1.85)),
    ):
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        fn(ax, data)
        fig.tight_layout(pad=0.35)
        for ext in ("pdf", "png"):
            p = OUT / f"{name}.{ext}"
            fig.savefig(p, dpi=600 if ext == "png" else None)
            written.append(p)
        plt.close(fig)
    return written


if __name__ == "__main__":
    d = load()
    print(f"loaded {len(d)} methods from {CSV.name}")
    print("h=16:", {k: round(v, 4) for k, v in at_horizon(d, 16).items()})
    for p in build():
        print("wrote", p.name)
