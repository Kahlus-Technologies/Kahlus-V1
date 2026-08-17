"""Figure: the overlap inflation law that explains the retracted result.

An earlier version of this work reported an MSE of 3.116 and r = 0.972 for
future-window forecasting. That number was an artifact. The input window and
the target window shared 126 of 127 samples, so the model was scored mostly on
copying its own input rather than on predicting anything.

The useful move is not to delete the number but to explain it, because the
mechanism is general and quantifiable. If a window of W samples is scored
against a target that shares all but H of them, the error is diluted by the
copied fraction, and the apparent improvement over an honest strictly-future
forecast is bounded by W/H, with equality when copying is free.

Panel (a) tests that bound on 286 configurations spanning 11 signals, real and
synthetic. Nothing lands above the identity line, which is what "bound" has to
mean to be worth stating. The retracted configuration is marked: W=127, H=1
predicts 127x, and 126.1x was what the repository actually measured.

Panel (b) shows the departure from the bound is not noise either. Cells fall
below W/H exactly when copying is not free, and the exact expression that
accounts for the measured copy error tracks them. Including AR(1), a random
walk, and white noise matters here: white noise cannot be forecast at all, so
its inflation is pure overlap arithmetic and nothing else.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import common as C

ARTIFACT = "overlap_law/overlap_law_results.json"


def build() -> None:
    C.rc()
    d = C.load(ARTIFACT)
    cells = d["cells"]

    obs = np.array([c["inflation_observed"] for c in cells], dtype=float)
    bound = np.array([c["inflation_bound_W_over_H"] for c in cells], dtype=float)
    exact = np.array([c["inflation_exact_with_measured_copy"] for c in cells],
                     dtype=float)
    ratio = np.array([c["copy_to_future_ratio"] for c in cells], dtype=float)
    synth = np.array([c["signal"] in ("white_noise", "ar1_0.5", "ar1_0.95",
                                      "random_walk") for c in cells])

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(C.COL_W, 3.9),
        gridspec_kw={"height_ratios": [1.0, 0.78]},
    )

    # ---- (a) observed vs bound --------------------------------------------
    lim = [0.8, max(bound.max(), obs.max()) * 1.6]
    ax.plot(lim, lim, color=C.C_BAD, lw=0.9, ls="--", zorder=1,
            label="bound $W/H$")
    ax.scatter(bound[~synth], obs[~synth], s=7, color=C.C_MODEL, alpha=0.75,
               lw=0, label="recorded signals", zorder=2)
    ax.scatter(bound[synth], obs[synth], s=7, color=C.C_RIDGE, alpha=0.75,
               lw=0, marker="^", label="synthetic", zorder=2)

    hb = d["historical_bug_reference"]
    ax.scatter([hb["predicted_bound"]], [hb["measured_in_repo"]], s=26,
               facecolor="none", edgecolor=C.C_BAD, lw=1.1, zorder=4)
    # The identity line runs through the whole panel, so the only clear space
    # is the region below it. Park the label there and point at the marker.
    ax.annotate(f"retracted result\nW={hb['W']}, H={hb['H']}: "
                f"{hb['measured_in_repo']:g}x",
                xy=(hb["predicted_bound"], hb["measured_in_repo"]),
                xytext=(0.97, 0.10), textcoords="axes fraction",
                ha="right", va="bottom", fontsize=C.PT_ANNOT, color=C.C_BAD,
                arrowprops=dict(arrowstyle="-", color=C.C_BAD, lw=0.6,
                                shrinkA=1.0, shrinkB=3.0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    # Plain decade labels, not mathtext. A 10^2 superscript is drawn at 0.7x
    # the tick size, so a 7 pt tick prints its exponent at 4.9 pt, under the
    # legibility floor. Writing the decades out keeps every glyph at 7 pt.
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(mticker.FixedLocator([1, 10, 100]))
        a.set_major_formatter(mticker.FixedFormatter(["1", "10", "100"]))
        a.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("predicted bound $W/H$")
    ax.set_ylabel("observed inflation")
    ax.grid(alpha=0.25, lw=0.4, which="major")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4,
              labelspacing=0.22, borderpad=0.1, handletextpad=0.5)
    n_viol = d["n_bound_violations"]
    ax.set_title(f"(a) {d['n_cells']} configurations, {n_viol} violations",
                 loc="left")

    # ---- (b) departure vs the correction term ------------------------------
    frac = obs / bound
    frac_exact = exact / bound
    order = np.argsort(ratio)
    # Same reason as panel (a): plot the decade exponent directly so the axis
    # needs no superscripts to be read.
    lr = np.log10(ratio)
    bx.scatter(lr, frac, s=7, color=C.C_MODEL, alpha=0.75, lw=0,
               label="observed", zorder=3)
    bx.plot(lr[order], frac_exact[order], color=C.C_BAD, lw=1.0,
            label="exact, with measured copy error", zorder=2)
    bx.set_xlabel("copy error / future error (log10)")
    bx.set_ylabel("observed / bound")
    bx.axhline(1.0, color=C.C_PERSIST, lw=0.7, ls=":")
    bx.grid(alpha=0.25, lw=0.4)
    bx.legend(loc="lower left", frameon=False, handlelength=1.4,
              labelspacing=0.22, borderpad=0.1)
    med = d["observed_over_bound"]["median"]
    # The median belongs inside the axes, not in the title: at 3.25 in the
    # title has room for the claim or the number, not both.
    bx.annotate(f"median {med:.3f}", xy=(0.62, 0.30), xycoords="axes fraction",
                ha="right", va="center", fontsize=C.PT_ANNOT)
    bx.set_title("(b) equality when copying is free", loc="left")

    manifest = {
        "artifact": ARTIFACT,
        "law": d["law"],
        "n_cells": d["n_cells"],
        "n_bound_violations": n_viol,
        "observed_over_bound": d["observed_over_bound"],
        "historical_bug_reference": hb,
        "n_synthetic_cells": int(synth.sum()),
    }
    p = C.save(fig, "fig_overlap_law", manifest)
    print(f"wrote {p}")
    print(f"  {d['n_cells']} cells, {n_viol} violations, "
          f"median obs/bound {med:.4f}")
    print(f"  retracted config: bound {hb['predicted_bound']:g}x, "
          f"measured {hb['measured_in_repo']:g}x")


if __name__ == "__main__":
    build()
