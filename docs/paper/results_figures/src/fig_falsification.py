"""Figure: the falsification suite behind the short-horizon claim.

A short-horizon win on autocorrelated timeseries is exactly the kind of result
that is usually an artifact, so the claim is worth more when the reader can see
what was done to try to break it. Each row is one attempt.

Panel (a) is the negative controls. Both must land at the variance of the
target, MSE = 1, and they do: shuffling the targets destroys the effect and
feeding noise destroys it, which is what rules out a harness that would score
well on anything.

Panel (b) is the robustness checks, plotted as the paired GRU-minus-ridge
advantage with its bootstrap CI. The headline number sits at the top. Below it,
the same quantity recomputed under conditions that a sceptic would demand:
non-overlapping windows at stride 384, so no two windows share a sample; eight
resampled subject splits; and a comparison against SGD-fit linear regression
rather than ridge, in case the baseline was simply mis-tuned. The effect has to
survive all of them, and the interval has to exclude zero in each.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import common as C

ARTIFACT = "forecast_crossover/adversarial_validation.json"
MAIN = "forecast_crossover/forecast_scaleup_results.json"
H = "16"  # the horizon the adversarial suite probes


def build() -> None:
    C.rc()
    d = C.load(ARTIFACT)
    main = C.load(MAIN)

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(C.COL_W, 3.5),
        gridspec_kw={"height_ratios": [0.72, 1.0]},
    )

    # ---- (a) negative controls --------------------------------------------
    nulls = d["t2_t3_nulls_h16"]
    names = ["shuffled\ntargets", "noise\ninput"]
    keys = ["shuffled_targets", "noise_input"]
    x = np.arange(len(keys))
    w = 0.36
    ridge_v = [nulls[k]["ridge_mse"] for k in keys]
    gru_v = [nulls[k]["gru_mse"] for k in keys]

    ax.bar(x - w / 2, ridge_v, w, color=C.C_RIDGE, alpha=0.85, label="ridge")
    ax.bar(x + w / 2, gru_v, w, color=C.C_MODEL, alpha=0.85, label="GRU")
    ax.axhline(1.0, color=C.C_BAD, lw=0.8, ls="--")
    ax.annotate("target variance (no skill)", xy=(0.98, 1.0),
                xycoords=("axes fraction", "data"), xytext=(0, 2),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=C.PT_ANNOT, color=C.C_BAD)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("MSE")
    ax.set_ylim(0, 1.28)
    ax.legend(loc="upper left", frameon=False, handlelength=1.2,
              labelspacing=0.2, borderpad=0.1, ncol=2, columnspacing=0.8)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_title("(a) negative controls collapse to no skill", loc="left")

    # ---- (b) robustness ----------------------------------------------------
    rows = []
    m = main["comparisons"][H]["gru_vs_ridge"]
    rows.append(("headline (overlapping windows)", m["mean_delta"],
                 m["ci95_low"], m["ci95_high"]))

    t7 = d["t7_non_overlapping"][H]
    rows.append((f"non-overlapping (stride {d['t7_non_overlapping']['stride']})",
                 t7["mean_delta"], t7["ci95_low"], t7["ci95_high"]))

    t6 = d["t6_split_robustness"][H]
    dl = np.asarray(t6["deltas"], dtype=float)
    rows.append((f"{t6['n_splits']} resampled splits",
                 float(dl.mean()), float(dl.min()), float(dl.max())))

    t4 = d["t4_linear_sgd"][H]["gru_vs_linear_sgd"]
    rows.append(("vs linear SGD baseline", t4["mean_delta"],
                 t4["ci95_low"], t4["ci95_high"]))

    y = np.arange(len(rows))[::-1]
    for yi, (_, mu, lo, hi) in zip(y, rows):
        ok = lo > 0
        col = C.C_GOOD if ok else C.C_BAD
        bx.plot([lo, hi], [yi, yi], color=col, lw=1.4, solid_capstyle="butt")
        bx.plot([mu], [yi], marker="o", ms=3.4, color=col, zorder=3)

    bx.axvline(0, color="black", lw=0.8)
    bx.set_yticks(y)
    bx.set_yticklabels([r[0] for r in rows], fontsize=C.PT_ANNOT)
    bx.set_xlabel("MSE(ridge) - MSE(GRU), 95% CI")
    bx.set_ylim(-0.6, len(rows) - 0.4)
    bx.grid(axis="x", alpha=0.25, lw=0.4)
    bx.set_title(f"(b) the h={H} advantage survives", loc="left")

    manifest = {
        "artifact": ARTIFACT,
        "main_artifact": MAIN,
        "dataset": main["data"],
        "horizon_samples": int(H),
        "nulls": {k: {"ridge_mse": nulls[k]["ridge_mse"],
                      "gru_mse": nulls[k]["gru_mse"]} for k in keys},
        "robustness": [
            {"label": r[0], "mean_delta": r[1], "lo": r[2], "hi": r[3]}
            for r in rows
        ],
        "index_audit": d["t1_index_audit"],
    }
    p = C.save(fig, "fig_falsification", manifest)
    print(f"wrote {p}")
    for lbl, mu, lo, hi in rows:
        print(f"  {lbl}: {mu:+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"{'excludes 0' if lo > 0 else 'INCLUDES 0'}")


if __name__ == "__main__":
    build()
