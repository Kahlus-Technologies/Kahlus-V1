"""Figures for the 2026-08-15 isolated-residual evidence run + tuned-ridge audit.

Regenerate:
  python3 artifacts/isolated_residual_evidence_20260815/make_figures.py

Reads only committed JSON in this directory; writes figures/ alongside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / ".claude/skills/scientific-figures/scripts"))
import figure_style as fs  # noqa: E402

HORIZONS = ("1", "8", "16", "32")
MS = {"1": 10, "8": 80, "16": 160, "32": 320}
HEADLINE_H = "16"

pack = json.load(open(HERE / "runs/isolated_residual_sleep/scores.json"))["folds"]
audit = json.load(open(HERE / "tuned_ridge_audit_result.json"))
subjects = sorted(audit)
assert set(subjects) == set(pack), "audit and pack folds must agree"

persist = {h: np.array([pack[s][h]["persist"] for s in subjects]) for h in HORIZONS}
weak = {h: np.array([audit[s][h]["pack_ridge"] for s in subjects]) for h in HORIZONS}
tuned = {h: np.array([audit[s][h]["tuned_ridge"] for s in subjects]) for h in HORIZONS}
stack = {h: np.array([audit[s][h]["pack_stack"] for s in subjects]) for h in HORIZONS}
n = len(subjects)


def boot_ci(per_fold: np.ndarray, reps: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = np.array([per_fold[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return per_fold.mean(), *np.percentile(means, [2.5, 97.5])


fs.use_paper_style()
fig, axes = fs.new_figure(cols=2, panels=3)
ax_sweep, ax_gap, ax_pair = axes

# --- Panel A: method sweep over horizon --------------------------------------
methods = [
    ("persistence", persist, fs.PALETTE[3], "o"),
    ("ridge, 16 lags $\\alpha{=}20$", weak, fs.PALETTE[1], "s"),
    ("ridge, 127 lags, tuned $\\alpha$", tuned, fs.PALETTE[2], "D"),
    ("GRU residual stack", stack, fs.PALETTE[0], "o"),
]
xs = [MS[h] for h in HORIZONS]
for name, data, color, marker in methods:
    m, lo, hi = zip(*(boot_ci(data[h]) for h in HORIZONS))
    m, lo, hi = np.array(m), np.array(lo), np.array(hi)
    ax_sweep.errorbar(xs, m, yerr=[m - lo, hi - m], marker=marker, markersize=4,
                      linewidth=1.5, capsize=2.5, color=color, label=name)
ax_sweep.set_xscale("log")
ax_sweep.set_xticks(xs, [str(x) for x in xs])
ax_sweep.set_xlabel("forecast horizon (ms)")
ax_sweep.set_ylabel("isolated test MSE (z-units, lower is better)")
ax_sweep.set_title("mean over 78 held-out subjects,\n95% subject-bootstrap CI")
ax_sweep.legend(fontsize=7, frameon=True, framealpha=0.95, loc="upper left")
ax_sweep.grid(alpha=0.3)

# --- Panel B (headline): does the gap survive the tuned baseline? ------------
rng = np.random.default_rng(1)
tick_labels = []
for i, h in enumerate(HORIZONS):
    gap = tuned[h] - stack[h]
    jitter = rng.uniform(-0.16, 0.16, n)
    ax_gap.scatter(i + jitter, gap, s=7, color=fs.PALETTE[0], alpha=0.35,
                   linewidths=0, zorder=2)
    m, lo, hi = boot_ci(gap)
    ax_gap.errorbar([i + 0.32], [m], yerr=[[m - lo], [hi - m]], fmt="o",
                    markersize=5, capsize=3, color="black", zorder=3)
    tick_labels.append(f"{MS[h]}\n{int((gap > 0).sum())}/{n}")
ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1)
ax_gap.annotate("no advantage", (0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color="0.25")
ax_gap.set_xticks(range(len(HORIZONS)), tick_labels)
ax_gap.set_xlabel("forecast horizon (ms)")
ax_gap.set_ylabel(r"$\Delta$MSE per subject: tuned ridge $-$ GRU stack")
ax_gap.set_title("GRU advantage after baseline tuning\n(points below dashes falsify the claim)")
ax_gap.grid(alpha=0.3, axis="y")

# --- Panel C: paired per-subject comparison at the headline horizon ----------
t16, s16 = tuned[HEADLINE_H], stack[HEADLINE_H]
lims = (min(t16.min(), s16.min()) * 0.85, max(t16.max(), s16.max()) * 1.15)
ax_pair.plot(lims, lims, color="black", linestyle="--", linewidth=1,
             label="identity (no advantage)")
ax_pair.scatter(t16, s16, s=10, color=fs.PALETTE[0], alpha=0.6, linewidths=0)
ax_pair.set_xscale("log")
ax_pair.set_yscale("log")
ax_pair.set_xlim(lims)
ax_pair.set_ylim(lims)
ax_pair.set_xlabel("tuned-ridge MSE per subject (160 ms)")
ax_pair.set_ylabel("GRU-stack MSE per subject (160 ms)")
below = int((s16 < t16).sum())
ax_pair.set_title(f"160 ms, per-subject pairs:\n{below}/{n} below the identity line")
ax_pair.legend(fontsize=7, frameon=True, framealpha=0.95, loc="upper left")
ax_pair.grid(alpha=0.3, which="both")

fig.suptitle(
    "Sleep-EDF LOSO, 78 subjects, one seed (preliminary):\n"
    "a nonlinear residual survives a per-fold-tuned 127-lag ridge at every horizon, 10–320 ms",
)

out = HERE / "figures"
out.mkdir(exist_ok=True)
fs.assert_no_clip(fig)
fs.save(fig, str(out / "residual_survives_tuned_ridge"))
print("wrote", out / "residual_survives_tuned_ridge.{pdf,png}")
