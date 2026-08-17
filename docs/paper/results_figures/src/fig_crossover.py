"""Figure: the crossover between the GRU and a per-horizon-tuned ridge.

This is the paper's central quantitative claim, and it is neither the win the
early draft claimed nor the null the current draft concedes. It is a crossover:
the learned model is better at short lead times and worse at long ones, and the
sign flip is the finding.

Two panels, stacked, because they answer two different objections:

  (a) The horizon curve, with mean-of-trace and persistence drawn in. Without
      those two references a reader cannot tell whether either model is doing
      anything at all, since at long horizons both converge on predicting the
      mean.

  (b) Per-subject paired deltas. The aggregate means in (a) could be driven by
      a handful of subjects, so the count of subjects favouring the GRU is
      drawn per horizon. At h=4 it is 27/27; at h=256 it is 0/27. A unanimous
      split in both directions is what makes the crossover a real effect rather
      than a noisy average.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import common as C

ARTIFACT = "forecast_crossover/forecast_scaleup_results.json"


def build() -> None:
    C.rc()
    d = C.load(ARTIFACT)

    fs = d["sampling_rate_hz"]
    hs = d["horizons"]
    ms = [h / fs * 1000.0 for h in hs]

    comp = d["comparisons"]
    gru = [comp[str(h)]["mean_mse"]["kahlus_gru"] for h in hs]
    ridge = [comp[str(h)]["mean_mse"]["ridge_ar"] for h in hs]
    persist = [comp[str(h)]["mean_mse"]["persistence"] for h in hs]
    mean_tr = [comp[str(h)]["mean_mse"]["mean_of_trace"] for h in hs]
    favor = [comp[str(h)]["gru_vs_ridge"]["n_subjects_favoring_reference"] for h in hs]
    n_sub = comp[str(hs[0])]["gru_vs_ridge"]["n_subjects"]

    # Per-subject deltas: positive means the GRU beat ridge on that subject.
    ps = d["per_subject_mse"]
    deltas = []
    for h in hs:
        r = ps["ridge_ar"][str(h)]
        g = ps["kahlus_gru"][str(h)]
        deltas.append([r[s] - g[s] for s in sorted(r)])

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(C.COL_W, 4.05),
        gridspec_kw={"height_ratios": [1.0, 0.92]},
    )

    # ---- (a) horizon curve -------------------------------------------------
    ax.plot(ms, mean_tr, color=C.C_MEAN, ls=":", marker="", label="mean of trace")
    ax.plot(ms, persist, color=C.C_PERSIST, ls="--", lw=1.0, label="persistence")
    ax.plot(ms, ridge, color=C.C_RIDGE, marker="s", ms=2.6, label="ridge AR (tuned)")
    ax.plot(ms, gru, color=C.C_MODEL, marker="o", ms=2.6, label="GRU")
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10", "100", "1000"])
    ax.set_xlabel("forecast horizon (ms, log)")
    ax.set_ylabel("MSE (normalised)")
    ax.grid(alpha=0.25, lw=0.4)

    # Mark the sign change. Found from the data, not asserted.
    sign = np.sign(np.array(ridge) - np.array(gru))
    idx = next((i for i in range(1, len(sign)) if sign[i] != sign[i - 1]), None)
    if idx is not None:
        xc = float(np.sqrt(ms[idx - 1] * ms[idx]))  # geometric midpoint on log axis
        ax.axvline(xc, color=C.C_BAD, lw=0.7, ls="-", alpha=0.7)
        ax.annotate(
            "crossover", xy=(xc, ax.get_ylim()[1]), xytext=(2.0, -1.0),
            textcoords="offset points", fontsize=C.PT_ANNOT, color=C.C_BAD,
            ha="left", va="top", rotation=90,
        )
    ax.legend(loc="upper left", frameon=False, handlelength=1.6,
              labelspacing=0.25, borderpad=0.1)
    ax.set_title("(a) GRU wins early, ridge wins late", loc="left")

    # ---- (b) per-subject paired deltas -------------------------------------
    pos = np.arange(len(hs))
    bp = bx.boxplot(deltas, positions=pos, widths=0.62, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=0.8))
    for i, box in enumerate(bp["boxes"]):
        # Colour by which model the subjects favour, so the flip is visible
        # even before reading the axis.
        box.set(facecolor=C.C_MODEL if np.median(deltas[i]) > 0 else C.C_RIDGE,
                alpha=0.55, lw=0.6)
    for part in ("whiskers", "caps"):
        for ln in bp[part]:
            ln.set_lw(0.6)

    bx.axhline(0, color="black", lw=0.7)
    bx.set_xticks(pos)
    bx.set_xticklabels([f"{m:g}" for m in ms], rotation=45, ha="right")
    bx.set_xlabel("forecast horizon (ms)")
    bx.set_ylabel("MSE(ridge) - MSE(GRU)")
    bx.grid(axis="y", alpha=0.25, lw=0.4)

    # Counts of subjects favouring the GRU, drawn inside the axes so they
    # cannot collide with the panel title. Headroom is added first.
    lo, hi = bx.get_ylim()
    bx.set_ylim(lo, hi + 0.20 * (hi - lo))
    ytxt = bx.get_ylim()[1] - 0.045 * (bx.get_ylim()[1] - lo)
    for i, k in enumerate(favor):
        bx.annotate(f"{k}", xy=(i, ytxt), ha="center", va="top",
                    fontsize=C.PT_ANNOT,
                    color=C.C_MODEL if k > n_sub / 2 else C.C_RIDGE)
    bx.set_title(f"(b) subjects favouring GRU (of {n_sub}), per horizon",
                 loc="left")

    manifest = {
        "artifact": ARTIFACT,
        "dataset": d["data"],
        "n_test_subjects": n_sub,
        "horizons_samples": hs,
        "horizons_ms": ms,
        "series": {
            "kahlus_gru": gru, "ridge_ar": ridge,
            "persistence": persist, "mean_of_trace": mean_tr,
        },
        "n_favoring_gru": favor,
        "crossover_between_ms": [ms[idx - 1], ms[idx]] if idx is not None else None,
    }
    p = C.save(fig, "fig_crossover", manifest)
    print(f"wrote {p}")
    print(f"  crossover between {ms[idx-1]:g} and {ms[idx]:g} ms")
    print(f"  favouring GRU: {favor}")


if __name__ == "__main__":
    build()
