"""Figure for the overlap inflation law.

Left: predicted bound W/H vs observed inflation, log-log, with the identity line.
Points sit on it and never above it -- the bound is one-sided by construction.
Right: how far a cell falls below the bound, as a function of the one quantity
that explains it, MSE_copy/MSE_future.

  python3 scripts/plot_overlap_law.py --results runs/overlap_law/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("runs/overlap_law/results.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/overlap_law/overlap_law.png"))
    args = ap.parse_args()

    r = json.loads(args.results.read_text(encoding="utf-8"))
    cells = r["cells"]
    bound = np.array([c["inflation_bound_W_over_H"] for c in cells])
    obs = np.array([c["inflation_observed"] for c in cells])
    ratio = np.array([c["copy_to_future_ratio"] for c in cells])
    signals = [c["signal"] for c in cells]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    lim = [1.5, 400]
    ax.plot(lim, lim, color="black", lw=1.2, ls="--", zorder=1,
            label="identity: observed = W/H")
    sc = ax.scatter(bound, obs, c=np.log10(np.clip(ratio, 1e-9, None)), cmap="viridis",
                    s=34, alpha=0.85, edgecolors="none", zorder=3)
    # The historical Kahlus bug, placed on the same axes.
    hb = r["historical_bug_reference"]
    ax.scatter([hb["predicted_bound"]], [hb["measured_in_repo"]], marker="*", s=340,
               facecolor="#d62728", edgecolor="black", lw=0.8, zorder=5,
               label=f"historical bug: W={hb['W']}, H={hb['H']}")
    ax.annotate(f"predicted {hb['predicted_bound']:.0f}×\nmeasured {hb['measured_in_repo']:.1f}×",
                xy=(hb["predicted_bound"], hb["measured_in_repo"]),
                xytext=(0.42, 0.20), textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#d62728"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted bound  $W/H$   (from the methods section alone)")
    ax.set_ylabel("observed inflation\n$\\mathrm{MSE}_{future}/\\mathrm{MSE}_{overlap}$")
    ax.set_title(f"{len(cells)} configurations, {len(set(signals))} signals, "
                 f"{r['n_bound_violations']} violations", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6, which="both")
    fig.colorbar(sc, ax=ax, label="$\\log_{10}(\\mathrm{MSE}_{copy}/\\mathrm{MSE}_{future})$")

    frac = obs / bound
    for name in sorted(set(signals)):
        m = np.array([s == name for s in signals])
        style = dict(s=30, alpha=0.8, edgecolors="none")
        if name.startswith(("white", "ar1", "random")):
            style.update(marker="^")
        bx.scatter(np.clip(ratio[m], 1e-9, None), frac[m], label=name, **style)
    bx.axhline(1.0, color="black", lw=1.2, ls="--")
    bx.set_xscale("log")
    bx.set_xlabel("$\\mathrm{MSE}_{copy}/\\mathrm{MSE}_{future}$  (how exactly the copy is reproduced)")
    bx.set_ylabel("observed / bound")
    bx.set_title("Every departure from the bound is explained by one term", fontsize=11)
    bx.set_ylim(0.2, 1.06)
    bx.grid(alpha=0.25, lw=0.6, which="both")
    bx.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
