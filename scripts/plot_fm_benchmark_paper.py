#!/usr/bin/env python3
"""Paper figures for the PR #80 EEG foundation-model benchmark, from the raw artifact only.

Reads docs/results/eeg_fm_benchmark_2026-08-24/eeg_fm_benchmark.json and
paired_statistics_kahlus_reference.json. No number is typed in by hand.

Figure A (fig_fm_benchmark.pdf): two panels.
  left  : isolated test MSE vs horizon, mean over 27 held-out subjects with the
          bootstrap 95% CI, for the incumbent ladder (persistence, tuned ridge, GBM,
          our GRU) and the best external pretrained condition per family
          (full fine-tune). The mean-predictor line (MSE = 1 in z-units) marks
          the ceiling every entrant hits by 320 ms.
  right : per-horizon paired Delta = best rival - ours (subject-level bootstrap CI,
          Holm-corrected), with the verdict printed under each horizon.

Figure B (fig_random_twin.pdf): pretrained vs random-initialised twin, test MSE,
  one point per (model, condition, horizon). Points on or below the identity line
  are cells where pretraining bought nothing. Count printed in the title.

Run: python3 scripts/plot_fm_benchmark_paper.py [--out docs/paper/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs/results/eeg_fm_benchmark_2026-08-24"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Blind labels for the paper. The model is "ours"; never the internal name.
LABEL = {
    "persistence": "persistence",
    "tuned_ridge_ar": "tuned ridge (127-lag AR)",
    "gbm": "GBM",
    "kahlus_gru": "ours (task-trained GRU)",
    "eegpt__full_ft": "EEGPT, full fine-tune",
    "labram_base__full_ft": "LaBraM-Base, full fine-tune",
    "cbramod__full_ft": "CBraMod, full fine-tune",
    "biot_prest16__full_ft": "BIOT, full fine-tune",
}
COLOR = {
    "persistence": "#d62728", "tuned_ridge_ar": "#2ca02c", "gbm": "#8c564b",
    "kahlus_gru": "#1f77b4", "eegpt__full_ft": "#9467bd",
    "labram_base__full_ft": "#ff7f0e", "cbramod__full_ft": "#17becf",
    "biot_prest16__full_ft": "#7f7f7f",
}
MARK = {
    "persistence": "o", "tuned_ridge_ar": "D", "gbm": "v", "kahlus_gru": "s",
    "eegpt__full_ft": "^", "labram_base__full_ft": "P", "cbramod__full_ft": "X",
    "biot_prest16__full_ft": "h",
}
FS_HZ = 100  # Sleep-EDF Fpz-Cz, 100 Hz; horizon in samples -> ms = 10 * h


def load():
    bench = json.loads((RES / "eeg_fm_benchmark.json").read_text())
    paired = json.loads((RES / "paired_statistics_kahlus_reference.json").read_text())
    return bench, paired


def series(bench, entrant_id):
    rows = [r for r in bench["results"] if r["entrant_id"] == entrant_id]
    rows.sort(key=lambda r: r["horizon"])
    h = np.array([r["horizon"] for r in rows])
    m = np.array([r["test_mse_mean"] for r in rows])
    lo = np.array([r["test_mse_ci95_low"] for r in rows])
    hi = np.array([r["test_mse_ci95_high"] for r in rows])
    return h, m, lo, hi


def fig_benchmark(bench, paired, out: Path):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.32})

    # left: ladder + best external
    for eid in ["persistence", "tuned_ridge_ar", "gbm", "eegpt__full_ft", "labram_base__full_ft",
                "cbramod__full_ft", "biot_prest16__full_ft", "kahlus_gru"]:
        h, m, lo, hi = series(bench, eid)
        if len(h) == 0:
            continue
        lw = 2.0 if eid == "kahlus_gru" else 1.1
        z = 5 if eid == "kahlus_gru" else 3
        ax.plot(h * 1000 / FS_HZ, m, marker=MARK[eid], ms=4, lw=lw, color=COLOR[eid],
                label=LABEL[eid], zorder=z)
        ax.fill_between(h * 1000 / FS_HZ, lo, hi, color=COLOR[eid], alpha=0.10, lw=0, zorder=1)
    ax.axhline(1.0, ls=":", color="k", lw=0.9)
    ax.text(11, 1.015, "predict-the-mean (MSE = 1)", fontsize=6.5, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 80, 160, 320, 640, 1280, 2560])
    ax.set_xticklabels(["10", "20", "40", "80", "160", "320", "640", "1280", "2560"])
    ax.set_xlabel("forecast horizon (ms)")
    ax.set_ylabel("isolated test MSE (z-units, lower is better)")
    ax.set_title("Sleep-EDF, 27 held-out subjects, 3 seeds\n(band: 95% subject-bootstrap CI)", fontsize=8.5)
    ax.legend(loc="lower right", ncol=1, frameon=True, framealpha=0.9, edgecolor="0.8", handlelength=1.6, fontsize=6.5)
    ax.set_ylim(0.2, 1.6)

    # right: paired delta vs best rival, Holm-corrected
    per_h = [b for b in paired["by_horizon"]]
    per_h.sort(key=lambda b: b["horizon"])
    xs, deltas, los, his, labels, colors = [], [], [], [], [], []
    for b in per_h:
        if not b["claim_eligible"]:
            continue
        ext = [c for c in b["comparisons"] if "random_twin" not in c["entrant_id"]
               and c["entrant_id"] not in ("persistence", "tuned_ridge_ar", "gbm")]
        # best rival = lowest test MSE among external pretrained entrants
        best = min(ext, key=lambda c: c["test_mse_mean"])
        xs.append(b["horizon"]); deltas.append(best["mean_delta"])
        los.append(best["ci95_low"]); his.append(best["ci95_high"])
        # Verdict rule mirrors VERIFIED_NUMBERS_2026-08-28: win = Holm-significant with
        # ours lower; tie = declared statistical tie; loss = Holm-significant with rival
        # lower; everything else is not significant.
        if best["significant_after_holm"] and best["mean_delta"] > 0:
            v, c = "win", "#1f77b4"
        elif best["significant_after_holm"] and best["mean_delta"] < 0:
            v, c = "loss", "#d62728"
        elif best["statistical_tie"]:
            v, c = "tie", "#7f7f7f"
        else:
            v, c = "n.s.", "#bbbbbb"
        labels.append(f"h={b['horizon']}\n{v}")
        colors.append(c)
    xs = np.array(xs); deltas = np.array(deltas); los = np.array(los); his = np.array(his)
    xpos = np.arange(len(xs))
    bx.axhline(0, color="k", lw=0.9, ls="--")
    bx.errorbar(xpos, deltas, yerr=[deltas - los, his - deltas], fmt="none", ecolor="k", elinewidth=1, capsize=2, zorder=2)
    bx.scatter(xpos, deltas, c=colors, s=40, zorder=3, edgecolor="k", lw=0.5)
    bx.set_xticks(xpos); bx.set_xticklabels(labels, fontsize=6.5)
    bx.set_ylabel(r"$\Delta$MSE: best pretrained rival $-$ ours", fontsize=8)
    bx.set_title("paired per subject, Holm-corrected\nover 8 claim-eligible horizons", fontsize=8.5)
    bx.text(0.02, 0.97, "above zero: ours has lower error", transform=bx.transAxes, fontsize=6.5, va="top")
    n_win = sum(1 for l in labels if l.endswith("win"))
    n_tie = sum(1 for l in labels if l.endswith("tie"))
    n_ns = sum(1 for l in labels if l.endswith("n.s."))
    n_loss = sum(1 for l in labels if l.endswith("loss"))
    bx.text(0.98, 0.03, f"{n_win} wins, {n_tie} ties, {n_ns} n.s., {n_loss} losses\ngate: claim NOT certified",
            transform=bx.transAxes, fontsize=6.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6))
    assert (n_win, n_tie, n_ns, n_loss) == (5, 2, 1, 0), (n_win, n_tie, n_ns, n_loss)

    fig.savefig(out / "fig_fm_benchmark.pdf"); fig.savefig(out / "fig_fm_benchmark.png")
    print("wrote", out / "fig_fm_benchmark.{pdf,png}", "| wins", n_win, "ties", n_tie)


def fig_twin(bench, out: Path):
    rows = {(r["entrant_id"], r["horizon"]): r for r in bench["results"]}
    fam = {"eegpt": ("EEGPT", "#9467bd", "^"), "labram_base": ("LaBraM-Base", "#ff7f0e", "P"),
           "cbramod": ("CBraMod", "#17becf", "X"), "biot_prest16": ("BIOT", "#7f7f7f", "h")}
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    n_total = n_twin_wins = 0
    for m, (name, col, mk) in fam.items():
        for cond, alpha in (("frozen", 0.45), ("full_ft", 1.0)):
            pre, twin = [], []
            for (eid, h), r in rows.items():
                if eid == f"{m}__{cond}":
                    t = rows.get((f"{m}__{cond}__random_twin", h))
                    if t is None:
                        continue
                    pre.append(r["test_mse_mean"]); twin.append(t["test_mse_mean"])
            pre = np.array(pre); twin = np.array(twin)
            n_total += len(pre); n_twin_wins += int((twin <= pre).sum())
            ax.scatter(pre, twin, color=col, marker=mk, s=28, alpha=alpha, edgecolor="k", lw=0.4,
                       label=f"{name}, {'full fine-tune' if cond=='full_ft' else 'frozen probe'}")
    lim = (0.25, 1.35)
    ax.plot(lim, lim, "k--", lw=0.9, label="identity: pretraining bought nothing")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("pretrained weights, test MSE")
    ax.set_ylabel("random-init twin, test MSE")
    ax.set_title(f"twin at or below pretrained: {n_twin_wins} of {n_total} cells", fontsize=9)
    ax.text(0.03, 0.97, "below line: random init\nbeats pretraining", transform=ax.transAxes, fontsize=7, va="top")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=6, handlelength=1.2, columnspacing=0.8)
    fig.savefig(out / "fig_random_twin.pdf"); fig.savefig(out / "fig_random_twin.png")
    print("wrote", out / "fig_random_twin.{pdf,png}", "| twin<=pre", n_twin_wins, "/", n_total)
    assert n_total == 72, f"expected 72 model x condition x horizon cells, got {n_total}"
    return n_twin_wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/paper/figures"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    bench, paired = load()
    fig_benchmark(bench, paired, out)
    n = fig_twin(bench, out)
    # Self-check against the verified ledger (docs/paper/VERIFIED_NUMBERS_2026-08-28.md).
    assert n == 25, f"VERIFIED_NUMBERS says 25 of 72 twin cells; recomputed {n}"
    print("self-check OK: 25/72 twin cells reproduced from raw artifact")


if __name__ == "__main__":
    main()
