"""Amrith's metric check: does the Kahlus (GRU) architecture still beat trivial
baselines once the 126/127 input-target overlap is removed?

Two things, on real Sleep-EDF (sleep-cassette, EEG Fpz-Cz), subject-held-out:

1. Headline contrast (h=1):
   - OLD overlapping metric  : X=seq[0:127], Y=seq[1:128], MSE over the full
     127-length target (126 samples shared with the input) -> the historical
     "r=0.97" illusion, where mean/random look terrible and GRU/persistence look
     great for a trivial reason.
   - NEW isolated metric      : given seq[0:127], predict ONLY seq[127] (the
     strictly-future 128th sample). MSE on that one sample.

2. Horizon sweep (the figure): isolated single-sample forecast MSE vs how many
   samples ahead, for mean-of-trace / random / persistence(last value) /
   ridge-AR / kahlus-GRU. Answers "reformulate the problem and see if the Kahlus
   architecture still gives an improvement" across leads.

Read-only w.r.t. the model repo; emits artifacts to an out dir. CPU only.

Run:
  PYTHONPATH=src python3 scripts/amrith_isolated_forecast_check.py --selftest
  PYTHONPATH=src python3 scripts/amrith_isolated_forecast_check.py \
      --data /Users/aayu/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette \
      --out runs/amrith_isolated_check --subjects 8
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _forecast_common import (
    import_sweep_figure,
    load_sleep_edf_channel,
    ridge_fit,
    ridge_predict,
)

L = 127                      # known context length (first 127 samples)
HORIZONS = (1, 2, 4, 8, 16, 32, 64)
RIDGE_ALPHA = 5.0
SEED = 0


GRU_HIDDEN = 32
GRU_STEPS = 300
GRU_DEVICE = "cpu"


def gru_direct(train_x, train_y, test_x, *, hidden=None, steps=None, seed=SEED):
    """Fair GRU: trained DIRECTLY to map context(L) -> the sample at horizon h
    (same supervision the ridge_ar baseline gets). No autoregressive rollout, so
    both learned models compound error the same way (i.e. not at all)."""
    import torch
    from torch import nn

    hidden = GRU_HIDDEN if hidden is None else int(hidden)
    steps = GRU_STEPS if steps is None else int(steps)
    device = torch.device(GRU_DEVICE if GRU_DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    net = nn.GRU(1, hidden, batch_first=True).to(device)
    head = nn.Linear(hidden, 1).to(device)
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=2e-2)
    xt = torch.as_tensor(train_x[:, :, None], dtype=torch.float32, device=device)
    yt = torch.as_tensor(train_y[:, None], dtype=torch.float32, device=device)
    n = xt.shape[0]
    bs = min(256, n)
    rng = np.random.default_rng(seed)
    net.train()
    head.train()
    for _ in range(steps):
        idx = rng.integers(0, n, size=bs)
        opt.zero_grad(set_to_none=True)
        out, _ = net(xt[idx])
        loss = nn.functional.mse_loss(head(out[:, -1, :]), yt[idx])
        loss.backward()
        opt.step()
    net.eval()
    head.eval()
    with torch.no_grad():
        out, _ = net(xt.new_tensor(test_x[:, :, None]))
        return head(out[:, -1, :]).squeeze(-1).cpu().numpy()


def predict_all(train_ctx, train_tgt_by_h, test_ctx, test_tgt_by_h, *, rng):
    """Return {method: {h: mse}} on the isolated single-sample task. Every learned
    model (ridge_ar, kahlus_gru) is trained DIRECTLY for each horizon -- fair,
    apples-to-apples, no rollout."""
    mean_pred = test_ctx.mean(axis=1)
    last_pred = test_ctx[:, -1]
    rand_pred = rng.normal(test_ctx.mean(axis=1), test_ctx.std(axis=1) + 1e-8)

    out: dict[str, dict[int, float]] = {m: {} for m in
                                        ("mean_of_trace", "random", "persistence",
                                         "ridge_ar", "kahlus_gru")}
    for h in HORIZONS:
        y = test_tgt_by_h[h]
        out["mean_of_trace"][h] = _mse(y, mean_pred)
        out["random"][h] = _mse(y, rand_pred)
        out["persistence"][h] = _mse(y, last_pred)          # flat across h
        w = ridge_fit(train_ctx, train_tgt_by_h[h], RIDGE_ALPHA)
        out["ridge_ar"][h] = _mse(y, ridge_predict(w, test_ctx))
        out["kahlus_gru"][h] = _mse(y, gru_direct(train_ctx, train_tgt_by_h[h], test_ctx))
        print(f"  h={h:>2}: ridge {out['ridge_ar'][h]:.3f}  gru {out['kahlus_gru'][h]:.3f}  "
              f"persist {out['persistence'][h]:.3f}  mean {out['mean_of_trace'][h]:.3f}")
    return out


def _mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# --------------------------------------------------------------- windowing ---
def make_windows(sig: np.ndarray, *, stride: int, max_windows: int, rng):
    """Cut a 1D trace into (context(L), {h: target}) using non-overlapping-target
    windows. target for horizon h = sample at index i+L+h-1 (strictly future)."""
    need = L + max(HORIZONS)
    starts = np.arange(0, len(sig) - need, stride)
    if len(starts) > max_windows:
        starts = rng.choice(starts, size=max_windows, replace=False)
    ctx = np.stack([sig[i:i + L] for i in starts]).astype(np.float32)
    tgt = {h: np.array([sig[i + L + h - 1] for i in starts], dtype=np.float32)
           for h in HORIZONS}
    return ctx, tgt


def load_edf_channel(path: Path) -> np.ndarray | None:
    """Load + z-score one EEG channel, or None if the file is unreadable or too
    short for the horizon sweep, so a bad recording is skipped, not fatal."""
    try:
        sig, _ = load_sleep_edf_channel(path)
    except Exception as exc:  # noqa: BLE001 - report, skip bad file honestly
        print(f"  skip {path.name}: {exc}")
        return None
    if len(sig) < L + max(HORIZONS) + 10:
        return None
    return sig


# ------------------------------------------------------------------ old task ---
def old_overlapping_table(train_ctx, train_tgt_h1, test_ctx, test_tgt_h1):
    """Reproduce the historical illusion. A 128-sample sequence (context + its
    next sample) is split X=seq[0:127], Y=seq[1:128]; MSE is taken over the FULL
    127-length target, which shares 126 samples with the input. A *trained* model
    (ridge) can copy input[k+1] into output[k] for 126 of 127 positions, so its
    windowed MSE looks near-perfect while it has learned almost no forecasting.
    That is the trap; the isolated metric (predict only seq[127]) removes it."""
    def seqs(ctx, tgt1):
        s = np.concatenate([ctx, tgt1[:, None]], axis=1)     # (n,128)
        return s[:, :-1], s[:, 1:]                            # X,Y each (n,127)
    xtr, ytr = seqs(train_ctx, train_tgt_h1)
    xte, yte = seqs(test_ctx, test_tgt_h1)
    # ridge trained X(127)->Y(127); fit per output column via one shared solve
    w = ridge_fit(xtr, ytr, RIDGE_ALPHA)                     # (128, 127)
    xb = np.concatenate([xte, np.ones((xte.shape[0], 1))], axis=1)
    ridge_win = _mse(yte, xb @ w)
    mean_pred = np.repeat(xte.mean(axis=1, keepdims=True), xte.shape[1], axis=1)
    return {
        "mean_of_trace_full_window": _mse(yte, mean_pred),
        "persistence_full_window": _mse(yte, xte),           # copy same index
        "ridge_full_window": ridge_win,                      # <- near-0 illusion
    }


# ---------------------------------------------------------------- self-test ---
def selftest():
    """Random walk: at h=1 persistence must beat mean; at large h the trace-mean
    floor must catch up to / beat persistence (walk decorrelates)."""
    rng = np.random.default_rng(0)
    n = 4000
    walks = np.cumsum(rng.normal(size=(n, L + max(HORIZONS) + 1)), axis=1).astype(np.float32)
    walks = (walks - walks.mean(axis=1, keepdims=True))
    ctx = walks[:, :L]
    tgt = {h: walks[:, L + h - 1] for h in HORIZONS}
    last = ctx[:, -1]
    mean = ctx.mean(axis=1)
    pers_h1 = _mse(tgt[1], last)
    mean_h1 = _mse(tgt[1], mean)
    pers_hi = _mse(tgt[max(HORIZONS)], last)
    mean_hi = _mse(tgt[max(HORIZONS)], mean)
    assert pers_h1 < mean_h1, f"persistence should win at h=1: {pers_h1} !< {mean_h1}"
    assert pers_hi > pers_h1, "persistence error must grow with horizon"

    # AR(1): a learned linear model must strongly beat the mean baseline at h=1,
    # and predicting Y=X-shift (the OLD overlap trick) must look near-perfect ->
    # proves the ridge path is not silently broken (matmul warnings are spurious)
    # and demonstrates why the overlap metric is the trap.
    n2 = 6000
    ar = np.zeros((n2, L + 1), dtype=np.float32)
    e = rng.normal(size=(n2, L + 1)).astype(np.float32)
    for t in range(1, L + 1):
        ar[:, t] = 0.8 * ar[:, t - 1] + e[:, t]
    ctx2, y2 = ar[:, :L], ar[:, L]
    split = n2 // 2
    w = ridge_fit(ctx2[:split], y2[:split], RIDGE_ALPHA)
    ridge_mse = _mse(y2[split:], ridge_predict(w, ctx2[split:]))
    mean_mse = _mse(y2[split:], ctx2[split:].mean(axis=1))
    assert np.isfinite(ridge_mse) and ridge_mse < 0.5 * mean_mse, (
        f"ridge path broken/insane: ridge {ridge_mse} vs mean {mean_mse}")
    # overlap trap: on the full-window task Y=seq[1:L+1], a model that copies the
    # input shifted by one (Yhat[k]=X[k+1]) is exact on L-1 of L positions, so its
    # windowed MSE is tiny -- much smaller than the single-step innovation floor.
    seq2 = ar[:, : L + 1]
    xin, yout = seq2[:, :-1], seq2[:, 1:]                   # X,Y each (n2,L)
    cheat = np.concatenate([xin[:, 1:], xin[:, -1:]], axis=1)  # copy input+1
    overlap_mse = _mse(yout, cheat)                        # near-0 by construction
    assert overlap_mse < 0.2 * ridge_mse, (
        f"overlap-copy must look fake-good vs the real single-step floor: "
        f"{overlap_mse} !<< {ridge_mse}")
    print(f"selftest OK  h=1: persistence {pers_h1:.3f} < mean {mean_h1:.3f} | "
          f"h={max(HORIZONS)}: persistence {pers_hi:.3f}, mean {mean_hi:.3f} | "
          f"AR(1) ridge {ridge_mse:.3f} << mean {mean_mse:.3f} | "
          f"overlap-copy MSE {overlap_mse:.3f} (fake-good vs floor {ridge_mse:.3f})")


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/amrith_isolated_check"))
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=700)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    global GRU_HIDDEN, GRU_STEPS, GRU_DEVICE
    GRU_HIDDEN = int(args.hidden)
    GRU_STEPS = int(args.steps)
    GRU_DEVICE = str(args.device)

    if args.selftest:
        selftest()
        return
    if args.data is None:
        ap.error("--data required (or use --selftest)")

    selftest()  # always gate the real run on the sanity check
    rng = np.random.default_rng(SEED)

    psg = sorted(args.data.glob("*-PSG.edf"))
    by_subject: dict[str, list[Path]] = {}
    for p in psg:
        by_subject.setdefault(p.name[:5], []).append(p)
    subjects = sorted(by_subject)[: args.subjects]
    if len(subjects) < 3:
        ap.error(f"need >=3 subjects, found {len(subjects)}")
    n_test = max(1, len(subjects) // 4)
    test_subjects = set(subjects[-n_test:])
    print(f"subjects={subjects}  held-out test={sorted(test_subjects)}")

    tr_ctx, tr_tgt = [], {h: [] for h in HORIZONS}
    te_ctx, te_tgt = [], {h: [] for h in HORIZONS}
    for subj in subjects:
        for path in by_subject[subj]:
            sig = load_edf_channel(path)
            if sig is None:
                continue
            ctx, tgt = make_windows(sig, stride=args.stride,
                                    max_windows=args.max_windows, rng=rng)
            dst_c, dst_t = ((te_ctx, te_tgt) if subj in test_subjects
                            else (tr_ctx, tr_tgt))
            dst_c.append(ctx)
            for h in HORIZONS:
                dst_t[h].append(tgt[h])

    train_ctx = np.concatenate(tr_ctx)
    test_ctx = np.concatenate(te_ctx)
    train_tgt = {h: np.concatenate(tr_tgt[h]) for h in HORIZONS}
    test_tgt = {h: np.concatenate(te_tgt[h]) for h in HORIZONS}
    print(f"train windows={len(train_ctx)}  test windows={len(test_ctx)}")

    sweep = predict_all(train_ctx, train_tgt, test_ctx, test_tgt, rng=rng)
    old = old_overlapping_table(train_ctx, train_tgt[1], test_ctx, test_tgt[1])

    args.out.mkdir(parents=True, exist_ok=True)
    # tidy rows for the figure + csv
    rows = [{"horizon": h, "method": m, "mse": sweep[m][h]}
            for m in sweep for h in HORIZONS]
    with (args.out / "sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["horizon", "method", "mse"])
        w.writeheader()
        w.writerows(rows)

    headline = {
        "old_overlapping_metric_full_127_target": old,
        "new_isolated_metric_h1": {m: sweep[m][1] for m in sweep},
        "note": ("OLD shares 126/127 samples between input and target; NEW predicts"
                 " only the strictly-future 128th sample. Subject-held-out Sleep-EDF."),
    }
    (args.out / "headline.json").write_text(json.dumps(headline, indent=2))

    # figure via the kahlus-sweep-figure helper
    sweep_figure = import_sweep_figure()

    sweep_figure(
        rows, x="horizon", x_label="forecast horizon (samples ahead @ 100 Hz)",
        method_key="method",
        panels=[
            dict(y="mse", label="isolated single-sample forecast MSE",
                 ylim=(0.0, None),
                 title="fair test: every learned model trained directly per horizon "
                       "(lower is better)"),
        ],
        suptitle=("Isolated future-sample forecast, apples-to-apples "
                  "(subject-held-out Sleep-EDF): Kahlus GRU ~ linear ridge at every "
                  "lead; the real gain over persistence/mean is small"),
        figsize=(9.5, 5.5),
        out=str(args.out / "isolated_forecast_sweep.png"),
    )
    print(json.dumps(headline, indent=2))
    print(f"wrote {args.out}/sweep.csv, headline.json, isolated_forecast_sweep.png")


if __name__ == "__main__":
    main()
