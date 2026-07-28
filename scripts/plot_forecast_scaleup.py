"""Figure for the powered isolated-forecast result.

Left: mean per-subject MSE vs horizon for every model, with the crossover band
shaded. Right: per-subject paired deltas (ridge - GRU) as a strip plot, so the
reader sees the 27 individual subjects behind each p-value rather than a bar.

  python3 scripts/plot_forecast_scaleup.py --results runs/forecast_scaleup/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {
    "persistence": ("#9a9a9a", "o", "persistence (last value)"),
    "ridge_ar": ("#1f77b4", "s", "ridge-AR (tuned per horizon)"),
    "kahlus_gru": ("#d62728", "^", "Kahlus GRU (tuned, 3 seeds)"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("runs/forecast_scaleup/results.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/forecast_scaleup/forecast_crossover.png"))
    args = ap.parse_args()

    r = json.loads(args.results.read_text(encoding="utf-8"))
    horizons = r["horizons"]
    fs = r["sampling_rate_hz"]
    ms = [1000.0 * h / fs for h in horizons]
    test_subj = r["split"]["test"]
    comp = r["comparisons"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for method, (color, marker, label) in STYLE.items():
        y = [comp[str(h)]["mean_mse"][method] for h in horizons]
        ax.plot(ms, y, marker=marker, color=color, label=label, lw=1.9, ms=5.5)

    # Crossover: last horizon the GRU wins -> first horizon it does not.
    wins = [h for h in horizons if comp[str(h)]["gru_vs_ridge"]["mean_delta"] > 0
            and comp[str(h)]["gru_vs_ridge"]["wilcoxon_p"] < 0.05]
    if wins:
        lo = 1000.0 * max(wins) / fs
        hi = 1000.0 * min([h for h in horizons if h > max(wins)] or [max(wins)]) / fs
        ax.axvspan(lo, hi, color="#ffd27f", alpha=0.45, zorder=0)
        ax.text(np.sqrt(lo * hi), ax.get_ylim()[1] * 0.06, "crossover",
                ha="center", fontsize=9, color="#8a5a00")

    ax.set_xscale("log")
    ax.set_xlabel("forecast horizon (ms, log scale)")
    ax.set_ylabel("MSE on the strictly-future sample\n(z-scored EEG)")
    ax.set_title(f"Isolated forecast, {len(test_subj)} held-out subjects", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)

    # Per-subject paired deltas. Positive = GRU lower error for that subject.
    for i, h in enumerate(horizons):
        d = np.array([r["per_subject_mse"]["ridge_ar"][str(h)][s]
                      - r["per_subject_mse"]["kahlus_gru"][str(h)][s] for s in test_subj])
        jitter = (np.random.default_rng(i).random(d.size) - 0.5) * 0.28
        color = "#d62728" if d.mean() > 0 else "#1f77b4"
        bx.scatter(np.full(d.size, i) + jitter, d, s=13, alpha=0.5, color=color,
                   edgecolors="none")
        bx.plot([i - 0.3, i + 0.3], [d.mean()] * 2, color="black", lw=2, zorder=5)

    bx.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
    bx.set_xticks(range(len(horizons)))
    bx.set_xticklabels([f"{m:g}" for m in ms], fontsize=8)
    bx.set_xlabel("forecast horizon (ms)")
    bx.set_ylabel("per-subject Δ MSE  (ridge − GRU)")
    bx.set_title("Each dot is one held-out subject; bar = mean", fontsize=11)
    bx.grid(alpha=0.25, lw=0.6, axis="y")
    bx.text(0.985, 0.965, "GRU better ↑\nridge better ↓", transform=bx.transAxes,
            ha="right", va="top", fontsize=8.5, color="#444")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
