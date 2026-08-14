#!/usr/bin/env python3
"""Isolated residual-field courtroom for the 24h A100 window.

Physics, in order:
  0 persist          last sample
  1 ridge            linear AR on flattened multi-channel context
  2 oscillator       ridge residual from narrowband phase/amp
  3 spatial GRU/SSM  leftover nonlinear / coupling
  4 NFC              latent field on leftover (optional)

Final prediction is always ridge + residual_head. If a head cannot beat
predict-zero on the residual, it is dropped. Isolated: given X[0:L]
predict Y[L+h-1], subject-held-out. Do not call scripts/slurm/train_a100.sh.

  PYTHONPATH=src python3 scripts/isolated_residual_a100.py --selftest
  PYTHONPATH=src python3 scripts/isolated_residual_a100.py \\
      --corpus bnci --device cuda --steps 8000 --out runs/isolated_residual
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _forecast_common import ridge_fit, ridge_predict  # noqa: E402
from analysis.build_bnci_ridge_tensors import (  # noqa: E402
    DEFAULT_CACHE,
    EEG_CHANNEL_NAMES,
    load_subject_eeg,
    zscore_train,
)

L = 127
HORIZONS = (1, 8, 16, 32)
RIDGE_ALPHA = 20.0
C3 = EEG_CHANNEL_NAMES.index("C3")
TRAIN_SUBJ = (1, 3, 4, 5, 7, 9)
VAL_SUBJ = (2,)
TEST_SUBJ = (6, 8)
BAND_HZ = (3.0, 6.0, 10.0, 20.0)


def mse(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean((a - b) ** 2))


def make_windows(sig: np.ndarray, *, stride: int, max_windows: int, rng) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """sig [T, C] -> ctx [N, L, C], tgt[h] [N, C]. Target is strictly future."""
    need = L + max(HORIZONS)
    starts = np.arange(0, sig.shape[0] - need, stride)
    if starts.size == 0:
        raise ValueError("trace shorter than context+horizon")
    if starts.size > max_windows:
        starts = rng.choice(starts, size=max_windows, replace=False)
    ctx = np.stack([sig[i : i + L] for i in starts]).astype(np.float32)
    tgt = {
        h: np.stack([sig[i + L + h - 1] for i in starts]).astype(np.float32) for h in HORIZONS
    }
    return ctx, tgt


def osc_features(ctx: np.ndarray, fs: float) -> np.ndarray:
    """Narrowband phase/amp from the last 64 samples. ctx [N, L, C] -> [N, F]."""
    tail = ctx[:, -64:, :]
    spec = np.fft.rfft(tail, axis=1)
    freqs = np.fft.rfftfreq(64, d=1.0 / fs)
    cols = []
    for hz in BAND_HZ:
        k = int(np.argmin(np.abs(freqs - hz)))
        bin_ = spec[:, k, :]
        cols.append(bin_.real)
        cols.append(bin_.imag)
    return np.concatenate(cols, axis=1).astype(np.float32)


def _torch_fit_seq(train_x, train_y, test_x, *, kind: str, hidden: int, steps: int, device: str, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    c = train_x.shape[-1]
    if kind == "gru":
        enc = nn.GRU(c, hidden, batch_first=True)
        head = nn.Linear(hidden, c)

        def step(xb):
            out, _ = enc(xb)
            return head(out[:, -1])
    elif kind == "ssm":
        from neurotwin.models.torch_models import TinySSMBaseline

        net = TinySSMBaseline(c, c, latent_dim=hidden, n_layers=2)

        def step(xb):
            return net(xb)[:, -1]
    else:
        raise ValueError(kind)

    params = list(enc.parameters()) + list(head.parameters()) if kind == "gru" else list(net.parameters())
    opt = torch.optim.AdamW(params, lr=2e-3)
    xt = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    n = xt.shape[0]
    bs = min(256, n)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, size=bs), device=device)
        opt.zero_grad(set_to_none=True)
        pred = step(xt[idx])
        loss = nn.functional.mse_loss(pred, yt[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        chunks = []
        te = torch.as_tensor(test_x, dtype=torch.float32, device=device)
        for i in range(0, te.shape[0], 512):
            chunks.append(step(te[i : i + 512]).cpu().numpy())
        return np.concatenate(chunks, axis=0)


def nfc_fit(train_x, train_y, test_x, *, hidden: int, steps: int, device: str, seed: int):
    import torch
    from torch import nn
    from neurotwin.models.nfc import NeuralFieldCompiler, NeuralFieldCompilerConfig

    torch.manual_seed(seed)
    c = train_x.shape[-1]
    model = NeuralFieldCompiler(
        {"eeg": c},
        {"eeg": c},
        NeuralFieldCompilerConfig(latent_dim=hidden, n_layers=1, use_uncertainty=False),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    xt = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    n = xt.shape[0]
    bs = min(64, n)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, size=bs), device=device)
        opt.zero_grad(set_to_none=True)
        pred = model({"eeg": xt[idx]}, "eeg")
        if pred.ndim == 3:
            pred = pred[:, -1]
        loss = nn.functional.mse_loss(pred, yt[idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        chunks = []
        te = torch.as_tensor(test_x, dtype=torch.float32, device=device)
        for i in range(0, te.shape[0], 64):
            pred = model({"eeg": te[i : i + 64]}, "eeg")
            if pred.ndim == 3:
                pred = pred[:, -1]
            chunks.append(pred.cpu().numpy())
        return np.concatenate(chunks, axis=0)


def evaluate(train_ctx, train_tgt, test_ctx, test_tgt, *, fs: float, args) -> dict:
    persist = test_ctx[:, -1]
    scores = {}
    for h in HORIZONS:
        y = test_tgt[h]
        k = min(args.ridge_lags, train_ctx.shape[1])
        flat_tr = train_ctx[:, -k:, :].reshape(train_ctx.shape[0], -1)
        flat_te = test_ctx[:, -k:, :].reshape(test_ctx.shape[0], -1)
        w = ridge_fit(flat_tr, train_tgt[h], args.ridge_alpha)
        ridge = ridge_predict(w, flat_te)
        residual_tr = train_tgt[h] - ridge_predict(w, flat_tr)
        residual_te = y - ridge
        row = {
            "persist": mse(y, persist),
            "ridge": mse(y, ridge),
            "ridge_c3": mse(y[:, C3] if y.ndim == 2 and y.shape[1] > C3 else y, ridge[:, C3] if ridge.ndim == 2 and ridge.shape[1] > C3 else ridge),
        }
        osc_w = ridge_fit(osc_features(train_ctx, fs), residual_tr, args.ridge_alpha)
        osc = ridge + ridge_predict(osc_w, osc_features(test_ctx, fs))
        row["ridge+osc"] = mse(y, osc)
        residual_after_osc_tr = train_tgt[h] - (ridge_predict(w, flat_tr) + ridge_predict(osc_w, osc_features(train_ctx, fs)))
        residual_after_osc_te = y - osc
        if "gru" in args.models:
            gru_res = _torch_fit_seq(
                train_ctx, residual_after_osc_tr, test_ctx,
                kind="gru", hidden=args.hidden, steps=args.steps, device=args.device, seed=args.seed,
            )
            row["ridge+osc+gru"] = mse(y, osc + gru_res)
            row["gru_residual_vs_zero"] = mse(residual_after_osc_te, gru_res)
        if "ssm" in args.models:
            ssm_res = _torch_fit_seq(
                train_ctx, residual_after_osc_tr, test_ctx,
                kind="ssm", hidden=args.hidden, steps=args.steps, device=args.device, seed=args.seed,
            )
            row["ridge+osc+ssm"] = mse(y, osc + ssm_res)
        if "nfc" in args.models:
            try:
                nfc_res = nfc_fit(
                    train_ctx, residual_after_osc_tr, test_ctx,
                    hidden=max(32, args.hidden // 2), steps=max(200, args.steps // 4),
                    device=args.device, seed=args.seed,
                )
                row["ridge+osc+nfc"] = mse(y, osc + nfc_res)
            except Exception as exc:  # isolated job must not die on NFC shape
                row["nfc_error"] = str(exc)
        scores[h] = row
        print(f"  h={h:>2}  persist {row['persist']:.4f}  ridge {row['ridge']:.4f}  "
              f"ridge+osc {row['ridge+osc']:.4f}"
              + (f"  +gru {row['ridge+osc+gru']:.4f}" if "ridge+osc+gru" in row else "")
              + (f"  +nfc {row['ridge+osc+nfc']:.4f}" if "ridge+osc+nfc" in row else ""),
              flush=True)
    return scores


def synthetic(n=400, t=800, c=3, fs=250.0, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(t)
    sine = 0.35 * np.sin(2 * np.pi * 10.0 * time / fs)
    sig = np.zeros((t, c), dtype=np.float32)
    noise = rng.normal(scale=0.25, size=(t, c)).astype(np.float32)
    for ch in range(c):
        for i in range(1, t):
            sig[i, ch] = 0.8 * sig[i - 1, ch] + sine[i] + noise[i, ch]
    return sig, fs


def load_bnci(args, rng):
    raw = {}
    for s in TRAIN_SUBJ + VAL_SUBJ + TEST_SUBJ:
        sig, fs = load_subject_eeg(args.cache, s)
        raw[s] = (sig, fs)
    train_cat = np.concatenate([raw[s][0] for s in TRAIN_SUBJ], axis=0)
    fs = raw[TRAIN_SUBJ[0]][1]
    parts = {}
    for split, subs in (("train", TRAIN_SUBJ), ("val", VAL_SUBJ), ("test", TEST_SUBJ)):
        ctxs, tgts = [], {h: [] for h in HORIZONS}
        for s in subs:
            sig = zscore_train(train_cat, raw[s][0])
            ctx, tgt = make_windows(sig, stride=args.stride, max_windows=args.max_windows, rng=rng)
            ctxs.append(ctx)
            for h in HORIZONS:
                tgts[h].append(tgt[h])
        parts[split] = (np.concatenate(ctxs), {h: np.concatenate(tgts[h]) for h in HORIZONS})
    return parts, fs


def selftest() -> int:
    rng = np.random.default_rng(0)
    sig, fs = synthetic()
    ctx, tgt = make_windows(sig, stride=8, max_windows=600, rng=rng)
    n = ctx.shape[0]
    cut = int(0.7 * n)
    args = argparse.Namespace(
        ridge_alpha=5.0, ridge_lags=16, models=[], hidden=32, steps=4, device="cpu", seed=0,
    )
    scores = evaluate(ctx[:cut], {h: tgt[h][:cut] for h in HORIZONS},
                      ctx[cut:], {h: tgt[h][cut:] for h in HORIZONS},
                      fs=fs, args=args)
    h1 = scores[1]
    h8 = scores[8]
    assert h8["ridge"] < h8["persist"], scores
    assert h1["persist"] < 0.5, h1
    print("selftest ok", json.dumps({"h1": h1, "h8": h8}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--corpus", choices=("bnci",), default="bnci")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=Path("runs/isolated_residual"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=4000)
    ap.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    ap.add_argument("--ridge-lags", type=int, default=16)
    ap.add_argument("--models", default="gru")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.selftest:
        return selftest()
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    parts, fs = load_bnci(args, rng)
    print(f"train {parts['train'][0].shape} test {parts['test'][0].shape} fs={fs} device={args.device}")
    scores = evaluate(parts["train"][0], parts["train"][1], parts["test"][0], parts["test"][1], fs=fs, args=args)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "isolated residual field; ridge then oscillator then nonlinear leftover",
        "split": {"train": TRAIN_SUBJ, "val": VAL_SUBJ, "test": TEST_SUBJ},
        "L": L,
        "horizons": HORIZONS,
        "scores": {str(h): v for h, v in scores.items()},
        "win_rule": "claim skill only if residual model beats ridge and residual-vs-zero < residual variance",
    }
    (args.out / "scores.json").write_text(json.dumps(payload, indent=2))
    print("wrote", args.out / "scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
