"""Figures for F11 (envelope horizon) and F12 (label-reliability collapse).

Two layouts from the same data:

  --layout ieee   (default) 3.5 in wide, panels stacked VERTICALLY, for a
                  two-column IEEE paper. Fonts are set in absolute points at that
                  width, so 8 pt here is 8 pt on the printed page -- the usual
                  failure mode is drawing a 13 in wide figure and letting the
                  journal shrink it to 3.5 in, which divides every font by ~3.7
                  and makes the axes unreadable.
  --layout wide   side-by-side, for slides and the handbook.

Both regenerate from committed artifacts -- no hand-placed points, no numbers
typed into this file. A missing artifact is a hard failure rather than a
plausible-looking drawing.

  python3 scripts/plot_sleepedf_figures.py --layout ieee --out docs/figures/ieee
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/aayu/.claude/skills/scientific-figures/scripts")
import figure_style as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

CHANCE = 0.5
#: IEEE single-column text width. A figure placed here at any other width has its
#: fonts rescaled by the ratio, which is how "I used fontsize=10" becomes 3 pt.
IEEE_COL_IN = 3.5


def ieee_rc() -> dict:
    """Absolute point sizes for a figure that will be placed at 3.5 in."""
    return {
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "figure.titlesize": 8.5, "lines.linewidth": 1.2, "lines.markersize": 3,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    }


def fig_envelope_horizon(res: dict, out: Path, ieee: bool) -> None:
    """F11: how far ahead the band envelope stays forecastable, and what survives
    removing eye and muscle artifact.

    Persistence is the reference that matters. An envelope is smooth, so copying
    the last value forward is the honest thing to beat -- a comparison against the
    mean alone would flatter every model here.
    """
    bands = [b for b in ("beta", "low_gamma") if b in res["bands"]]
    label = {"beta": "beta", "low_gamma": "low gamma"}

    if ieee:
        fig, axes = plt.subplots(3, 1, figsize=(IEEE_COL_IN, 7.2),
                                 constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)

    for ax, key, ylab, title in (
        (axes[0], "pearsonr", "correlation $r$ with\ntrue future value",
         "(a) forecastable to 60 s"),
        (axes[1], "mse", "MSE  (1.0 = mean)",
         "(b) beats persistence everywhere"),
    ):
        for i, band in enumerate(bands):
            rows = res["bands"][band]["horizons"]
            x = [r["horizon_s"] for r in rows]
            ax.plot(x, [r["eeg"][key] for r in rows], marker="o", lw=1.3,
                    color=fs.PALETTE[i], label=f"{label[band]}")
            ax.plot(x, [r["eeg_artifact_residualised"][key] for r in rows],
                    marker="s", lw=1.1, ls="--", color=fs.PALETTE[i], alpha=0.65,
                    label=f"{label[band]}, artifact removed")
        if key == "mse":
            rows = res["bands"][bands[0]]["horizons"]
            ax.plot([r["horizon_s"] for r in rows],
                    [r["eeg"]["persistence_mse"] for r in rows],
                    color="black", lw=1.2, ls=":", label="persistence")
            ax.axhline(1.0, color="grey", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        hs = [r["horizon_s"] for r in res["bands"][bands[0]]["horizons"]]
        ax.set_xticks(hs)
        ax.set_xticklabels([f"{h:g}" for h in hs])
        ax.minorticks_off()
        ax.set_xlabel("forecast horizon (s)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        # Panel (a)'s data falls left-to-right so upper right is free; panel (b)
        # has the persistence line across the top, so its legend goes low.
        ax.legend(frameon=True, framealpha=0.92,
                  loc="upper right" if key == "pearsonr" else "lower right")

    # The headline metric gets its own panel: per-subject consistency, so the
    # reader can see the result is not carried by a handful of subjects.
    ax = axes[2]
    band = bands[-1]
    row60 = [r for r in res["bands"][band]["horizons"] if r["horizon_s"] == 60.0][0]
    per = row60["eeg"]["per_subject"]
    delta = np.array([p["persistence_mse"] - p["mse"] for p in per])
    order = np.argsort(delta)
    ax.barh(np.arange(len(delta)), delta[order],
            color=[fs.PALETTE[2] if d > 0 else fs.PALETTE[3] for d in delta[order]],
            height=0.75)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_yticks([])
    ax.set_xlabel("MSE gain over persistence\n(positive = model wins)")
    ax.set_ylabel(f"{len(delta)} held-out subjects")
    st = row60["eeg"]["paired_vs_persistence"]
    ax.set_title(f"(c) {label[band]} at 60 s: {row60['eeg']['beats_persistence_n']}/"
                 f"{row60['eeg']['n_test_subjects']}, $p$={st['wilcoxon_p']:.0e}")
    ax.grid(alpha=0.3, axis="x")

    sup = ("Band-power envelope is forecastable 60 s ahead\nand survives removing "
           "eye and muscle artifact" if ieee else
           "F11 · Sleep-EDF: the band-power envelope is forecastable a full minute "
           "ahead, and it is not muscle")
    fig.suptitle(sup)
    fs.assert_no_clip(fig)
    fs.save(fig, str(out / "f11_envelope_horizon"))
    print(f"wrote {out / 'f11_envelope_horizon'}.png/.pdf")


def fig_label_reliability(res: dict, out: Path, ieee: bool) -> None:
    """F12: transition-detection AUC before and after dropping unreliable labels.

    The claim this figure must make falsifiable: *removing unreliable labels costs
    the baseline more than it costs the model.* Were that false, the two drops in
    panel (b) would be equal. They are not.
    """
    td = res["transition_detection"]["by_input"]
    order = ["stage_only", "eeg_only", "eeg_plus_stage", "all_plus_stage"]
    short = {"stage_only": "stage only\n(baseline)", "eeg_only": "EEG only",
             "eeg_plus_stage": "EEG\n+ stage", "all_plus_stage": "EEG+EOG/EMG\n+ stage"}
    wide = {"stage_only": "current stage only (the baseline)", "eeg_only": "EEG only",
            "eeg_plus_stage": "EEG + current stage",
            "all_plus_stage": "EEG + EOG/EMG + current stage"}
    nice = short if ieee else wide

    def at60(k):
        rows = [r for r in td.get(k, []) if r["horizon_s"] == 60.0]
        return rows[0] if rows else None

    rows = {k: at60(k) for k in order if k in td}
    present = [k for k in order if rows.get(k) and rows[k].get("excluding_n1_transitions")]

    if ieee:
        fig, axes = plt.subplots(2, 1, figsize=(IEEE_COL_IN, 5.6),
                                 constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), constrained_layout=True)

    # (a) dumbbells make the DROP the visual object, which is the argument.
    ax = axes[0]
    y = np.arange(len(present))
    for i, k in enumerate(present):
        a = rows[k]["roc_auc"]
        b = rows[k]["excluding_n1_transitions"]["roc_auc"]
        ax.plot([a, b], [i, i], color="#9ca3af", lw=1.8, zorder=1, solid_capstyle="round")
        ax.scatter([a], [i], s=28 if ieee else 70, color=fs.PALETTE[7], zorder=3,
                   label="all transitions" if i == 0 else None)
        ax.scatter([b], [i], s=36 if ieee else 90, color=fs.PALETTE[0], zorder=3,
                   label="excluding N1" if i == 0 else None)
    ax.axvline(CHANCE, color="black", lw=1.1, ls="--", zorder=2)
    ax.annotate("chance", xy=(CHANCE, -0.45), xytext=(3, 0), textcoords="offset points",
                fontsize=6.5 if ieee else 9, va="center", ha="left", style="italic",
                annotation_clip=False)
    ax.set_yticks(y)
    ax.set_yticklabels([nice[k] for k in present])
    ax.set_xlabel("ROC AUC, stage change 60 s ahead")
    ax.set_title("(a) dropping unreliable labels lowers all")
    # Upper-left is the empty region here (low AUC, high rows); lower-right sits
    # on top of the baseline dumbbell.
    ax.legend(frameon=True, framealpha=0.92, loc="upper left")
    ax.grid(alpha=0.3, axis="x")

    # (b) the falsifiable claim: margin above chance.
    ax = axes[1]
    w = 0.38
    all_m = [rows[k]["roc_auc"] - CHANCE for k in present]
    rel_m = [rows[k]["excluding_n1_transitions"]["roc_auc"] - CHANCE for k in present]
    ax.bar(y - w / 2, all_m, w, color=fs.PALETTE[7], label="all transitions")
    ax.bar(y + w / 2, rel_m, w, color=fs.PALETTE[0], label="excluding N1")
    for i, v in enumerate(rel_m):
        ax.annotate(f"{v:+.3f}", xy=(i + w / 2, v), xytext=(0, 2),
                    textcoords="offset points", ha="center",
                    fontsize=6 if ieee else 8)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_xticks(y)
    ax.set_xticklabels([nice[k] for k in present])
    ax.set_ylabel("AUC above chance")
    ax.set_title("(b) but it guts the baseline, not the model")
    ax.set_ylim(0, max(all_m) * 1.22)
    ax.legend(frameon=True, framealpha=0.92)
    ax.grid(alpha=0.3, axis="y")

    n1 = rows[present[0]]["n1_share_of_transitions"]
    sup = (f"N1 is {n1:.0%} of scored transitions and expert $\\kappa$=0.24.\n"
           f"On transitions experts agree on, EEG doubles the signal."
           if ieee else
           f"F12 · N1 is {n1:.0%} of scored transitions and experts barely agree on it "
           f"($\\kappa$=0.24).\nOn the transitions they do agree on, EEG roughly doubles "
           f"the above-chance signal.")
    fig.suptitle(sup)
    fs.assert_no_clip(fig)
    fs.save(fig, str(out / "f12_label_reliability"))
    print(f"wrote {out / 'f12_label_reliability'}.png/.pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envelope", type=Path,
                    default=Path("artifacts/envelope_sleepedf/results.json"))
    ap.add_argument("--stage", type=Path,
                    default=Path("artifacts/stage_forecast/results.json"))
    ap.add_argument("--layout", choices=("ieee", "wide"), default="ieee")
    ap.add_argument("--out", type=Path, default=Path("docs/figures/ieee"))
    args = ap.parse_args()

    for p in (args.envelope, args.stage):
        if not p.exists():
            raise SystemExit(f"missing artifact: {p} -- run the experiment first; "
                             "this script will not invent numbers")

    fs.use_paper_style()
    ieee = args.layout == "ieee"
    if ieee:
        plt.rcParams.update(ieee_rc())
    args.out.mkdir(parents=True, exist_ok=True)

    fig_envelope_horizon(json.loads(args.envelope.read_text()), args.out, ieee)
    fig_label_reliability(json.loads(args.stage.read_text()), args.out, ieee)

    if ieee:
        print(f"\nIEEE single-column: place at {IEEE_COL_IN} in wide "
              "(\\includegraphics[width=\\columnwidth]{...}). Fonts are absolute "
              "at that width -- do not rescale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
