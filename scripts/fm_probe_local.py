"""Frontier EEG foundation model vs simple baselines on the isolated forecast task.

Same-data, same-task, subject-held-out (LOSO) courtroom, extended to judge a
pretrained foundation model by a *linear probe*: freeze the model, read its
embedding of the context window, fit a ridge readout to the strictly-future
target. Every method below predicts the SAME target sample from the SAME 1 s
context, so the comparison is honest.

Task (all at 200 Hz so the FM's native patch size is respected):
  context  = 200 samples (1 s) ending at t
  target   = one strictly-future sample at t + h, h in {10,80,160,320} ms
  methods  = persistence | tuned ridge (raw context) | CBraMod frozen probe
  eval     = leave-one-subject-out; MSE per subject; mean + per-fold win counts

Run (smoke):  PYTHONPATH=scripts /tmp/fm_probe_venv/bin/python scripts/fm_probe_local.py \
                  --max-subjects 6 --max-windows 600
Full:         ... --max-subjects 0 --max-windows 1200 --models cbramod
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _forecast_common import load_sleep_edf_channel, ridge_fit, ridge_predict  # noqa: E402

FS = 200.0
CTX = 200                       # 1 s context at 200 Hz = one CBraMod patch
H_MS = (10, 80, 160, 320)
H_SAMP = {ms: int(round(ms * FS / 1000)) for ms in H_MS}   # {2,16,32,64}
RIDGE_LAGS = 127
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
DATA = Path("/Users/aayu/datasets/sleep-edfx/sleep-cassette")
CACHE = Path(__file__).resolve().parents[1] / "artifacts/fm_probe/embeddings"


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def subject_groups() -> dict[str, list[Path]]:
    g: dict[str, list[Path]] = {}
    for p in sorted(DATA.glob("*-PSG.edf")):
        g.setdefault(p.name[3:5], []).append(p)
    return g


def make_windows(sig200: np.ndarray, *, stride: int, max_windows: int, rng):
    need = CTX + max(H_SAMP.values())
    starts = np.arange(0, sig200.shape[0] - need, stride)
    if starts.size > max_windows:
        starts = np.sort(rng.choice(starts, size=max_windows, replace=False))
    ctx = np.stack([sig200[s:s + CTX] for s in starts]).astype(np.float32)
    tgt = {ms: np.array([sig200[s + CTX - 1 + H_SAMP[ms]] for s in starts], np.float32)
           for ms in H_MS}
    return ctx, tgt


def load_subject(paths: list[Path], *, stride: int, max_windows: int, rng):
    ctxs, tgts = [], {ms: [] for ms in H_MS}
    for p in paths:
        sig, _ = load_sleep_edf_channel(p)                       # ~100 Hz, z-scored
        sig200 = resample_poly(sig, 2, 1).astype(np.float32)     # -> 200 Hz
        c, t = make_windows(sig200, stride=stride, max_windows=max_windows, rng=rng)
        ctxs.append(c)
        for ms in H_MS:
            tgts[ms].append(t[ms])
    return np.concatenate(ctxs), {ms: np.concatenate(tgts[ms]) for ms in H_MS}


def cbramod_embeddings(ctx: np.ndarray, *, device: str, batch: int = 256,
                       pretrained: bool = True):
    """Frozen CBraMod embedding of each [1 x 200] context window.

    pretrained=False is the null control: identical architecture, random
    weights (fixed seed). It separates "TUEG pretraining helps" from "any
    frozen nonlinear projection of the window helps"."""
    import torch
    from braindecode.models import CBraMod

    key = "_model_pre" if pretrained else "_model_rand"
    if not hasattr(cbramod_embeddings, key):
        if pretrained:
            m = CBraMod.from_pretrained("braindecode/cbramod-pretrained",
                                        return_encoder_output=True)
        else:
            torch.manual_seed(0)
            m = CBraMod(n_chans=1, n_times=CTX, sfreq=int(FS),
                        return_encoder_output=True)
        setattr(cbramod_embeddings, key, m.to(device).eval())
    model = getattr(cbramod_embeddings, key)
    out = []
    with torch.no_grad():
        for i in range(0, ctx.shape[0], batch):
            xb = torch.as_tensor(ctx[i:i + batch, None, :], device=device)  # [B,1,200]
            emb = model(xb)
            emb = emb if torch.is_tensor(emb) else emb[0]
            out.append(emb.reshape(emb.shape[0], -1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def tuned_ridge_pred(train_ctx, train_y, val_ctx, val_y, test_ctx):
    """127-lag ridge on the raw context, alpha picked on a held-out val split."""
    def feats(c):
        return c[:, -RIDGE_LAGS:]
    best_a, best_v = ALPHAS[0], np.inf
    for a in ALPHAS:
        w = ridge_fit(feats(train_ctx), train_y, a)
        v = mse(val_y, ridge_predict(w, feats(val_ctx)))
        if v < best_v:
            best_a, best_v = a, v
    xtr = np.concatenate([train_ctx, val_ctx])
    ytr = np.concatenate([train_y, val_y])
    w = ridge_fit(feats(xtr), ytr, best_a)
    return ridge_predict(w, feats(test_ctx)), best_a


def probe_pred(train_emb, train_y, val_emb, val_y, test_emb):
    """Linear probe with the SAME fairness the baseline ridge gets: features
    standardized on train stats, alpha tuned on the val split."""
    mu, sd = train_emb.mean(0), train_emb.std(0) + 1e-8
    ztr, zva, zte = (train_emb - mu) / sd, (val_emb - mu) / sd, (test_emb - mu) / sd
    best_a, best_v = ALPHAS[0], np.inf
    for a in ALPHAS:
        w = ridge_fit(ztr, train_y, a)
        v = mse(val_y, ridge_predict(w, zva))
        if v < best_v:
            best_a, best_v = a, v
    xtr = np.concatenate([ztr, zva])
    ytr = np.concatenate([train_y, val_y])
    w = ridge_fit(xtr, ytr, best_a)
    return ridge_predict(w, zte), best_a


def selftest() -> int:
    # AR(1) synthetic: a linear ridge must beat persistence at short horizon.
    rng = np.random.default_rng(0)
    x = np.zeros(20000, np.float32)
    for i in range(1, len(x)):
        x[i] = 0.9 * x[i - 1] + rng.normal(0, 0.4)
    c, t = make_windows(x, stride=13, max_windows=1500, rng=rng)
    n = len(c); cut = int(0.7 * n)
    pred, _ = tuned_ridge_pred(c[:cut], t[80][:cut], c[cut:], t[80][cut:], c[cut:])
    persist = c[cut:, -1]
    assert mse(t[80][cut:], pred) < mse(t[80][cut:], persist), "ridge must beat persistence on AR(1)"
    assert H_SAMP == {10: 2, 80: 16, 160: 32, 320: 64}, H_SAMP
    print("selftest ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--models", default="cbramod")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--max-subjects", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=600)
    ap.add_argument("--out", type=Path, default=Path("artifacts/fm_probe/scores.json"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rng = np.random.default_rng(0)
    groups = subject_groups()
    subjects = sorted(groups)
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]

    CACHE.mkdir(parents=True, exist_ok=True)
    data, emb = {}, {}
    for s in subjects:
        t = time.time()
        ctx, tgt = load_subject(groups[s], stride=args.stride, max_windows=args.max_windows, rng=rng)
        data[s] = (ctx, tgt)
        if "cbramod" in models:
            cf = CACHE / f"cbramod_{s}_{ctx.shape[0]}.npy"
            emb[s] = np.load(cf) if cf.exists() else cbramod_embeddings(ctx, device=args.device)
            if not cf.exists():
                np.save(cf, emb[s])
        if "cbramod-rand" in models:
            cf = CACHE / f"cbramodrand_{s}_{ctx.shape[0]}.npy"
            emb.setdefault("rand", {})
            emb["rand"][s] = np.load(cf) if cf.exists() else cbramod_embeddings(
                ctx, device=args.device, pretrained=False)
            if not cf.exists():
                np.save(cf, emb["rand"][s])
        print(f"subject {s}: {ctx.shape[0]} windows  ({time.time()-t:.1f}s)", flush=True)

    folds = {}
    for held in subjects:
        train = [s for s in subjects if s != held]
        val = train[-1]
        fit = train[:-1]
        te_ctx, te_tgt = data[held]
        tr_ctx = np.concatenate([data[s][0] for s in fit])
        va_ctx = data[val][0]
        row = {}
        for ms in H_MS:
            y_te = te_tgt[ms]
            tr_y = np.concatenate([data[s][1][ms] for s in fit])
            va_y = data[val][1][ms]
            r = {"persist": mse(y_te, te_ctx[:, -1])}
            pred, r["ridge_alpha"] = tuned_ridge_pred(tr_ctx, tr_y, va_ctx, va_y, te_ctx)
            r["tuned_ridge"] = mse(y_te, pred)
            if "cbramod" in models:
                tr_e = np.concatenate([emb[s] for s in fit])
                tr_ey = np.concatenate([data[s][1][ms] for s in fit])
                pred, r["cbramod_alpha"] = probe_pred(tr_e, tr_ey, emb[val], va_y, emb[held])
                r["cbramod_probe"] = mse(y_te, pred)
            if "cbramod-rand" in models:
                tr_e = np.concatenate([emb["rand"][s] for s in fit])
                tr_ey = np.concatenate([data[s][1][ms] for s in fit])
                pred, _ = probe_pred(tr_e, tr_ey, emb["rand"][val], va_y, emb["rand"][held])
                r["cbramod_rand_probe"] = mse(y_te, pred)
            row[str(ms)] = r
        folds[held] = row
        print(f"fold {held}: " + "  ".join(
            f"{ms}ms persist {row[str(ms)]['persist']:.3f} ridge {row[str(ms)]['tuned_ridge']:.3f}"
            + (f" cbramod {row[str(ms)]['cbramod_probe']:.3f}" if 'cbramod_probe' in row[str(ms)] else "")
            for ms in H_MS), flush=True)

    means = {}
    for ms in H_MS:
        keys = [k for k in ("persist", "tuned_ridge", "cbramod_probe", "cbramod_rand_probe")
                if k in folds[subjects[0]][str(ms)]]
        means[str(ms)] = {k: float(np.mean([folds[s][str(ms)][k] for s in subjects])) for k in keys}
    payload = {"task": "isolated forecast, 200 Hz, 1 s context, LOSO",
               "horizons_ms": list(H_MS), "n_subjects": len(subjects),
               "models": models, "folds": folds, "mean_over_folds": means}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print("\nMEAN OVER FOLDS:")
    for ms in H_MS:
        print(f"  {ms:>3}ms  " + "  ".join(f"{k} {v:.4f}" for k, v in means[str(ms)].items()))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
