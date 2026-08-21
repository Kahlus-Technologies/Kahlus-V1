"""Context-length x channel-count capability sweep for isolated EEG forecasting.

The unified courtroom feeds every model 1 s of one channel. Sleep EEG's
predictable machinery (spindles, K-complexes, slow waves) lives at 10-30 s
timescales, so this sweep measures what longer memory and a second channel
actually buy, at the horizons where headroom exists (80-2560 ms).

Configs: context in {1 s, 15 s, 30 s} x channels in {1, 3}. Methods per config:
  persist            last sample of the target channel
  tuned ridge        multi-scale lag features (dense last 1 s + strided
                     history), alpha tuned on a held-out train subject
  gru stack          GRU on the ridge residual; context re-sampled to a
                     mixed-rate sequence (history at 25 Hz + last 1 s at
                     200 Hz) so 30 s stays a short sequence
All targets are strictly-future samples of Fpz-Cz. Subject-held-out LOSO on a
fixed 24-subject subset for the scan; the winning config graduates to the full
78 x 3-seed protocol.

  PYTHONPATH=scripts /tmp/fm_probe_venv/bin/python scripts/fm_context_sweep.py --selftest
  ... fm_context_sweep.py --max-subjects 24 --out artifacts/fm_probe/context_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _forecast_common import ridge_fit, ridge_predict  # noqa: E402

FS = 200.0
H_MS = (80, 320, 1280, 2560)
H_SAMP = {ms: int(round(ms * FS / 1000)) for ms in H_MS}      # 16..512
CONTEXTS = (200, 3000, 6000)                                   # 1 s, 15 s, 30 s
CHANNEL_SETS = {
    1: ["EEG Fpz-Cz"],
    3: ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
}
TARGET_CH = 0                                                  # Fpz-Cz
DENSE = 200                                                    # dense tail (1 s)
STRIDE_HIST = 8                                                # history lag stride
HIST_RATE = 8                                                  # 200/8 = 25 Hz history for the GRU
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
DATA = Path(os.environ.get("KAHLUS_SLEEP_ROOT",
                           "/Users/aayu/datasets/sleep-edfx/sleep-cassette"))


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def load_multi(path: Path, channels, *, skip=1_000_000, take=300_000):
    """[T, C] float32 at 200 Hz, each channel z-scored per recording."""
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    picks = [c for c in channels if c in raw.ch_names]
    assert picks[0] == channels[0], f"target channel missing in {path.name}"
    raw.pick(picks)
    raw.load_data(verbose="ERROR")
    sig = raw.get_data().astype(np.float64)[:, skip:skip + take]         # [C, T] 100 Hz
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-8)
    up = np.stack([resample_poly(ch, 2, 1) for ch in sig])               # 200 Hz
    return up.T.astype(np.float32)                                       # [T, C]


def make_windows(sig, ctx_len, *, stride, max_windows, rng):
    need = ctx_len + max(H_SAMP.values())
    starts = np.arange(0, sig.shape[0] - need, stride)
    if starts.size > max_windows:
        starts = np.sort(rng.choice(starts, size=max_windows, replace=False))
    ctx = np.stack([sig[s:s + ctx_len] for s in starts]).astype(np.float32)  # [N, L, C]
    tgt = {ms: np.array([sig[s + ctx_len - 1 + H_SAMP[ms], TARGET_CH] for s in starts],
                        np.float32) for ms in H_MS}
    return ctx, tgt


def ridge_feats(ctx):
    """Dense last second + strided history, all channels, flattened."""
    dense = ctx[:, -DENSE:, :]
    hist = ctx[:, :-DENSE:STRIDE_HIST, :] if ctx.shape[1] > DENSE else None
    parts = [dense.reshape(len(ctx), -1)]
    if hist is not None and hist.shape[1] > 0:
        parts.insert(0, hist.reshape(len(ctx), -1))
    return np.concatenate(parts, axis=1)


def gru_seq(ctx):
    """Mixed-rate sequence: history at 25 Hz then the last second at 200 Hz."""
    dense = ctx[:, -DENSE:, :]
    if ctx.shape[1] > DENSE:
        hist = ctx[:, :-DENSE:HIST_RATE, :]
        return np.concatenate([hist, dense], axis=1)
    return dense


def tuned_ridge(xf_tr, y_tr, xf_va, y_va, xf_te):
    best_a, best_v = ALPHAS[0], np.inf
    for a in ALPHAS:
        w = ridge_fit(xf_tr, y_tr, a)
        v = mse(y_va, ridge_predict(w, xf_va))
        if v < best_v:
            best_a, best_v = a, v
    w = ridge_fit(np.concatenate([xf_tr, xf_va]), np.concatenate([y_tr, y_va]), best_a)
    return ridge_predict(w, xf_te), (lambda x: ridge_predict(w, x)), best_a


def nl_residual(train_seq, train_res, test_seq, *, seed, device, hidden=64,
                steps=1500, batch=192):
    """Causal-TCN residual head (repo-canonical TorchTCNBaseline). A TCN is
    parallel over time, so 900-token mixed-rate sequences train ~30x faster
    than a GRU on MPS; the sweep's question — does long context feed a
    nonlinear head — is head-agnostic, and the head choice is recorded in the
    output payload."""
    import torch
    from torch import nn

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from neurotwin.models.baselines import TorchTCNBaseline

    torch.manual_seed(seed)
    c = train_seq.shape[-1]
    enc = TorchTCNBaseline(c, hidden, hidden_dim=hidden, kernel_size=5).to(device)
    head = nn.Linear(hidden, 1).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=2e-3)
    xt = torch.as_tensor(train_seq, dtype=torch.float32, device=device)
    yt = torch.as_tensor(train_res[:, None], dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    n = xt.shape[0]
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, size=min(batch, n)), device=device)
        opt.zero_grad(set_to_none=True)
        out = enc(xt[idx])
        loss = nn.functional.mse_loss(head(out[:, -1]), yt[idx])
        loss.backward()
        opt.step()
    preds = []
    te = torch.as_tensor(test_seq, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(0, te.shape[0], 512):
            out = enc(te[i:i + 512])
            preds.append(head(out[:, -1]).cpu().numpy()[:, 0])
    return np.concatenate(preds)


def subject_groups():
    g = {}
    for p in sorted(DATA.glob("*-PSG.edf")):
        g.setdefault(p.name[3:5], []).append(p)
    return g


def selftest() -> int:
    rng = np.random.default_rng(0)
    t = np.arange(40000)
    sig = (np.sin(2 * np.pi * 1.0 * t / FS) +
           0.3 * rng.normal(size=t.size)).astype(np.float32)[:, None]
    sig = np.concatenate([sig, sig * 0.5 + 0.1 * rng.normal(size=(t.size, 1)).astype(np.float32)], axis=1)
    for ctx_len in (200, 3000):
        c, tg = make_windows(sig, ctx_len, stride=37, max_windows=800, rng=rng)
        assert c.shape[1:] == (ctx_len, 2) and len(tg[2560]) == len(c)
        f = ridge_feats(c)
        assert f.shape[1] == (min(ctx_len, DENSE) * 2 +
                              (max(0, (ctx_len - DENSE) + STRIDE_HIST - 1) // STRIDE_HIST) * 2), \
            (ctx_len, f.shape)
        s = gru_seq(c)
        assert s.shape[1] == min(ctx_len, DENSE) + (0 if ctx_len <= DENSE else
                                                    -(-(ctx_len - DENSE) // HIST_RATE)), \
            (ctx_len, s.shape)
    # a 1 Hz sine is exactly predictable from 15 s of context at 2560 ms lead:
    c, tg = make_windows(sig, 3000, stride=37, max_windows=800, rng=rng)
    cut = int(0.7 * len(c))
    pred, _, _ = tuned_ridge(ridge_feats(c[:cut]), tg[2560][:cut],
                             ridge_feats(c[cut:]), tg[2560][cut:], ridge_feats(c[cut:]))
    persist = c[cut:, -1, TARGET_CH]
    assert mse(tg[2560][cut:], pred) < 0.5 * mse(tg[2560][cut:], persist), \
        "long-context ridge must beat persistence on a slow sine"
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-subjects", type=int, default=24)
    ap.add_argument("--max-windows", type=int, default=500)
    ap.add_argument("--stride", type=int, default=400)
    ap.add_argument("--gru-steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    ap.add_argument("--channel-sets", default="1,3")
    ap.add_argument("--out", type=Path, default=Path("artifacts/fm_probe/context_sweep.json"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    contexts = [int(x) for x in args.contexts.split(",")]
    chsets = [int(x) for x in args.channel_sets.split(",")]
    groups = subject_groups()
    subjects = sorted(groups)[: args.max_subjects]

    # Calibrate before committing: measure the real GRU step rate on this
    # device and print the total estimate up front. A run may never again be
    # a black box with a wrong ETA.
    rng0 = np.random.default_rng(0)
    cal_seq = rng0.normal(size=(256, 600, 2)).astype(np.float32)
    cal_res = rng0.normal(size=256).astype(np.float32)
    t0 = time.time()
    nl_residual(cal_seq, cal_res, cal_seq[:32], seed=0, device=args.device, steps=50)
    rate = 50 / (time.time() - t0)
    n_train = len(contexts) * len(chsets) * len(subjects) * len(H_MS) * args.seeds
    est_h = n_train * args.gru_steps / rate / 3600
    print(f"calibration: {rate:.0f} TCN steps/s -> {n_train} trainings x "
          f"{args.gru_steps} steps ~= {est_h:.1f} h (+ ridge/loading overhead)", flush=True)

    partial_path = args.out.with_suffix(".partial.json")
    results = {}
    for nch in chsets:
        chans = CHANNEL_SETS[nch]
        raw = {}
        for s in subjects:
            raw[s] = [load_multi(p, chans) for p in groups[s]]
        for ctx_len in contexts:
            key = f"ctx{ctx_len}_ch{nch}"
            rng = np.random.default_rng(0)
            data = {}
            for s in subjects:
                cs, ts = [], {ms: [] for ms in H_MS}
                for sig in raw[s]:
                    c, tg = make_windows(sig, ctx_len, stride=args.stride,
                                         max_windows=args.max_windows, rng=rng)
                    cs.append(c)
                    for ms in H_MS:
                        ts[ms].append(tg[ms])
                data[s] = (np.concatenate(cs),
                           {ms: np.concatenate(ts[ms]) for ms in H_MS})
            folds = {}
            t0 = time.time()
            for held in subjects:
                train = [s for s in subjects if s != held]
                val, fit = train[-1], train[:-1]
                te_ctx, te_tgt = data[held]
                tr_ctx = np.concatenate([data[s][0] for s in fit])
                va_ctx = data[val][0]
                xtr, xva, xte = ridge_feats(tr_ctx), ridge_feats(va_ctx), ridge_feats(te_ctx)
                str_tr, str_va, str_te = gru_seq(tr_ctx), gru_seq(va_ctx), gru_seq(te_ctx)
                row = {}
                for ms in H_MS:
                    y_te = te_tgt[ms]
                    y_tr = np.concatenate([data[s][1][ms] for s in fit])
                    y_va = data[val][1][ms]
                    r = {"persist": mse(y_te, te_ctx[:, -1, TARGET_CH])}
                    pred_te, predict, r["alpha"] = tuned_ridge(xtr, y_tr, xva, y_va, xte)
                    r["tuned_ridge"] = mse(y_te, pred_te)
                    res_tr = np.concatenate([y_tr, y_va]) - predict(np.concatenate([xtr, xva]))
                    stack = []
                    for seed in range(args.seeds):
                        g = nl_residual(np.concatenate([str_tr, str_va]), res_tr, str_te,
                                         seed=seed, device=args.device, steps=args.gru_steps)
                        stack.append(mse(y_te, pred_te + g))
                    r["nl_stack"] = float(np.mean(stack))
                    row[str(ms)] = r
                folds[held] = row
                results[key] = {"folds": folds}
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text(json.dumps(results, indent=1))
                print(f"  {key} fold {held} done ({len(folds)}/{len(subjects)})", flush=True)
            means = {str(ms): {k: float(np.mean([folds[s][str(ms)][k] for s in subjects]))
                               for k in ("persist", "tuned_ridge", "nl_stack")}
                     for ms in H_MS}
            results[key] = {"folds": folds, "mean_over_folds": means}
            print(f"{key} done in {time.time()-t0:.0f}s:", flush=True)
            for ms in H_MS:
                m = means[str(ms)]
                print(f"  {ms:>4}ms persist {m['persist']:.4f} ridge {m['tuned_ridge']:.4f} "
                      f"stack {m['nl_stack']:.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"task": "context x channel capability sweep, isolated forecast, LOSO",
         "contexts": contexts, "channel_sets": chsets, "horizons_ms": list(H_MS),
         "n_subjects": len(subjects), "nonlinear_head": "causal_tcn_k5_h64", "results": results}, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
