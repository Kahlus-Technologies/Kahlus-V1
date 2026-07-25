"""Raw-EEG waveform diagram: what ridge sees vs what it predicts, old task vs new.

Amrith asked (twice) for example diagrams of raw EEG waveforms from the benchmark
with the ridge input and prediction target labeled. This draws exactly that on a
real Sleep-EDF trace, and contrasts the two task framings on the same window:

  Panel A (OLD, overlap-inflated): X = seq[0:127] is the ridge input, Y = seq[1:128]
    is the target. They share 126 of 127 samples (shaded). The "prediction" is
    almost a copy, which is why the old MSE looked near-perfect.

  Panel B (NEW, honest): the same 127-sample window is the input; the target is
    only the single strictly-future sample (seq[127]). Ridge's prediction of that
    one point is marked against the truth.

Run:
  PYTHONPATH=src python3 scripts/amrith_waveform_diagram.py --selftest
  PYTHONPATH=src python3 scripts/amrith_waveform_diagram.py \
      --data /Users/aayu/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette \
      --out runs/amrith_waveform
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _forecast_common import ar1_sequences, load_sleep_edf_channel, ridge_fit, ridge_predict

L = 127
RIDGE_ALPHA = 5.0
SEED = 0


def build_windows(sig, stride=256):
    starts = np.arange(0, len(sig) - (L + 1), stride)
    ctx = np.stack([sig[i:i + L] for i in starts]).astype(np.float32)   # (n,127)
    nxt = np.array([sig[i + L] for i in starts], dtype=np.float32)      # (n,) the 128th
    return ctx, nxt


def selftest():
    # AR(1): ridge predicts the next sample far better than the trace mean.
    ar = ar1_sequences(4000, L + 1, coef=0.8, seed=0)
    ctx, nxt = ar[:, :L], ar[:, L]
    sp = ar.shape[0] // 2
    w = ridge_fit(ctx[:sp], nxt[:sp], RIDGE_ALPHA)
    mse_ridge = float(np.mean((nxt[sp:] - ridge_predict(w, ctx[sp:])) ** 2))
    mse_mean = float(np.mean((nxt[sp:] - ctx[sp:].mean(axis=1)) ** 2))
    assert mse_ridge < 0.6 * mse_mean, f"ridge next-sample path broken: {mse_ridge} vs {mse_mean}"
    print(f"selftest OK  AR(1) next-sample ridge {mse_ridge:.3f} < mean {mse_mean:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/amrith_waveform"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.data is None:
        ap.error("--data required (or --selftest)")
    selftest()

    psg = sorted(args.data.glob("*-PSG.edf"))
    if not psg:
        ap.error(f"no PSG EDFs in {args.data}")
    sig, ch = load_sleep_edf_channel(psg[0], take=400_000)
    ctx, nxt = build_windows(sig)
    # train ridge on next-sample, hold out an example window to display
    sp = int(len(ctx) * 0.8)
    w = ridge_fit(ctx[:sp], nxt[:sp], RIDGE_ALPHA)

    # pick a visually clean held-out example (moderate amplitude, not flat)
    cand = np.arange(sp, len(ctx))
    stds = ctx[cand].std(axis=1)
    ex = cand[np.argsort(np.abs(stds - np.median(stds)))[0]]
    x_ex = ctx[ex]
    y_ex = float(nxt[ex])
    yhat = float(ridge_predict(w, x_ex[None, :])[0])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.0, 5.0), constrained_layout=True)
    t = np.arange(L + 1)

    # ---- Panel A: OLD overlapping task ----
    seq = np.concatenate([x_ex, [y_ex]])           # 128 samples
    axA.plot(t, seq, color="0.4", lw=1.0, zorder=1)
    axA.axvspan(1, L - 1, color="tab:red", alpha=0.12, zorder=0,
                label="126/127 samples shared by input & target")
    axA.plot(t[:L], seq[:L], marker="o", ms=2.5, lw=0, color="tab:blue",
             label="ridge INPUT  X = seq[0:127]")
    axA.plot(t[1:], seq[1:], marker="x", ms=3, lw=0, color="tab:orange",
             label="ridge TARGET Y = seq[1:128] (shifted by 1)")
    axA.set_title("OLD task: target is the input shifted by one sample\n"
                  "-> 126/127 overlap, 'prediction' is nearly a copy (MSE looks ~0)")
    axA.set_xlabel("sample index (100 Hz)")
    axA.set_ylabel(f"{ch} (z-scored)")
    axA.legend(fontsize=7, loc="upper right")
    axA.grid(alpha=0.3)

    # ---- Panel B: NEW isolated task ----
    axB.plot(t[:L], x_ex, color="tab:blue", lw=1.2, marker="o", ms=2.5,
             label="ridge INPUT  (127 known samples)")
    axB.axvline(L - 0.5, color="0.6", ls=":", lw=1.0)
    axB.plot([L], [y_ex], marker="o", ms=9, color="tab:green",
             label=f"TRUE future sample  seq[127] = {y_ex:+.2f}")
    axB.plot([L], [yhat], marker="D", ms=9, color="tab:red",
             label=f"ridge PREDICTION = {yhat:+.2f}")
    axB.annotate("", xy=(L, yhat), xytext=(L - 8, x_ex[-1]),
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.0))
    axB.set_title("NEW task: predict ONLY the strictly-future 128th sample\n"
                  "-> no overlap; this is the honest single-step forecast")
    axB.set_xlabel("sample index (100 Hz)")
    axB.set_ylabel(f"{ch} (z-scored)")
    axB.legend(fontsize=7, loc="upper right")
    axB.grid(alpha=0.3)

    fig.suptitle("What goes into ridge and what it predicts, on a real Sleep-EDF window "
                 "(left: overlap-inflated old task; right: honest single-step task)",
                 fontsize=11)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "amrith_waveform_diagram.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"example window index {ex}: true={y_ex:+.3f} ridge={yhat:+.3f} "
          f"|err|={abs(y_ex - yhat):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
