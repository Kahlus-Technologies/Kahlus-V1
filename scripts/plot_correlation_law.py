"""Figure for the correlation inflation law (F10).

Left: observed correlation-gap deflation against the two candidate bounds. The
exact bound holds everywhere; the W/H shortcut carried over from F9 does not,
and the points that break it sit above the identity line where a violation has
to appear.

Right: the failure is not noise. Rearranging the exact bound gives, with no free
parameters,

    (exact - 1) / (W/H - 1) = (s_pred * s_targ)^-1/2

so every cell must lie on one curve. It does, to machine precision.

  python3 scripts/plot_correlation_law.py --results runs/correlation_law/results.json
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
    ap.add_argument("--results", type=Path, default=Path("runs/correlation_law/results.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/correlation_law/correlation_law.png"))
    args = ap.parse_args()

    r = json.loads(args.results.read_text(encoding="utf-8"))
    c = r["cells"]
    obs = np.array([x["gap_deflation_observed"] for x in c])
    exact = np.array([x["gap_bound_exact"] for x in c])
    wh = np.array([x["gap_bound_W_over_H"] for x in c])
    sp = np.array([x["pred_var_ratio_future_over_copy"] for x in c])
    st = np.array([x["targ_var_ratio_future_over_copy"] for x in c])
    signals = [x["signal"] for x in c]
    n_wh_bad = int(r["n_W_over_H_violations"])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    lim = [1.2, 700]
    ax.plot(lim, lim, color="black", lw=1.2, ls="--", zorder=1, label="identity: observed = bound")
    ax.scatter(wh, obs, marker="x", s=26, lw=1.0, color="#d62728", alpha=0.75, zorder=3,
               label=f"W/H shortcut — {n_wh_bad}/{len(c)} violated")
    ax.scatter(exact, obs, s=30, alpha=0.85, edgecolors="none", color="#1f77b4", zorder=4,
               label=f"exact bound $1/(1-\\mu)$ — 0/{len(c)} violated")
    ax.fill_between(lim, lim, [lim[1]] * 2, color="#d62728", alpha=0.06, zorder=0)
    ax.text(1.6, 300, "bound violated\nin this region", fontsize=8.5, color="#d62728",
            style="italic", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted bound on correlation-gap deflation")
    ax.set_ylabel("observed\n$(1-r_{future})\\,/\\,(1-r_{overlap})$")
    ax.set_title("F9's $W/H$ does not carry over to correlation", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25, lw=0.6, which="both")

    # The collapse. (exact-1)/(W/H-1) = (s_pred*s_targ)^-1/2, exactly.
    s = sp * st
    y = (exact - 1.0) / (wh - 1.0)
    xs = np.logspace(np.log10(s.min() * 0.7), np.log10(s.max() * 1.4), 200)
    bx.plot(xs, xs ** -0.5, color="black", lw=1.3, ls="--", zorder=1,
            label="$(s_{pred}\\,s_{targ})^{-1/2}$  (no fitted parameter)")
    for name in sorted(set(signals)):
        m = np.array([q == name for q in signals])
        style = dict(s=30, alpha=0.8, edgecolors="none")
        if name.startswith(("white", "ar1", "random")):
            style.update(marker="^")
        bx.scatter(s[m], y[m], label=name, **style)
    bx.axvline(1.0, color="grey", lw=0.9, ls=":", zorder=1)
    bx.annotate("$W/H$ exact only here", xy=(1.0, 1.0), xytext=(0.30, 0.62),
                fontsize=8.5, color="dimgrey",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="dimgrey"))
    # The sharpest cell on the plot: nothing is forecastable, yet r reads 0.98.
    wn = min((x for x in c if x["signal"] == "white_noise"),
             key=lambda x: x["pred_var_ratio_future_over_copy"] * x["targ_var_ratio_future_over_copy"])
    wn_x = wn["pred_var_ratio_future_over_copy"] * wn["targ_var_ratio_future_over_copy"]
    bx.annotate(f"white noise, $W$={wn['W']} $H$={wn['H']}:\n"
                f"reported $r={wn['r_overlap']:.2f}$, true $r_{{future}}={wn['r_future']:.2f}$\n"
                f"$W/H$ understates by {wn['gap_bound_exact'] / wn['gap_bound_W_over_H']:.1f}$\\times$",
                xy=(wn_x, (wn["gap_bound_exact"] - 1) / (wn["gap_bound_W_over_H"] - 1)),
                xytext=(0.46, 0.86), textcoords="axes fraction",
                fontsize=8.5, color="#1f4e79", style="italic",
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#1f4e79"))
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlabel("$s_{pred}\\cdot s_{targ}$   (future/copy per-position variance ratio)")
    bx.set_ylabel("excess over $W/H$\n$(\\mathrm{exact}-1)/(W/H-1)$")
    bx.set_title("The whole departure is prediction shrinkage", fontsize=11)
    bx.grid(alpha=0.25, lw=0.6, which="both")
    bx.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
