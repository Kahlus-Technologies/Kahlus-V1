"""Tuned-ridge audit of the 2026-08-15 sleep LOSO evidence pack.

Replicates the pack's exact protocol (raw Fpz-Cz, skip 1M take 300k, stride 128,
L=127, train-fold-only scalar z-score) and asks one question: how much of the
GRU stack's gap over the pack's 16-lag alpha=20 ridge survives against a
127-lag ridge with alpha tuned per fold+horizon on held-out TRAIN subjects?

F26 precedent: an undertuned linear baseline manufactured a fake GRU win once
already. This is the same audit pointed at our own result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/aayu/Workspace/developer/Kahlus-V1")
PACK = Path("/tmp/evidence/evidence-kahlus-isolated-residual-20260815")
DATA = Path("/Users/aayu/datasets/sleep-edfx/sleep-cassette")
OUT = Path("/tmp/tuned_ridge_audit_result.json")

L = 127
HORIZONS = (1, 8, 16, 32)
STRIDE = 128
SKIP, TAKE = 1_000_000, 300_000
ALPHAS = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
N_VAL_SUBJ = 6


def load_raw_channel(path: Path) -> np.ndarray:
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    ch = "EEG Fpz-Cz" if "EEG Fpz-Cz" in raw.ch_names else next(
        (c for c in raw.ch_names if "EEG" in c), raw.ch_names[0])
    raw.pick([ch])
    raw.load_data(verbose="ERROR")
    return raw.get_data()[0].astype(np.float64)[SKIP:SKIP + TAKE].astype(np.float32)


def windows(sig: np.ndarray):
    need = L + max(HORIZONS)
    starts = np.arange(0, sig.shape[0] - need, STRIDE)
    ctx = np.stack([sig[i:i + L] for i in starts]).astype(np.float32)
    tgt = {h: np.array([sig[i + L + h - 1] for i in starts], dtype=np.float32) for h in HORIZONS}
    return ctx, tgt


def ridge_solve(x, y, alpha):
    xb = np.concatenate([x, np.ones((x.shape[0], 1), np.float32)], axis=1)
    a = xb.T @ xb + alpha * np.eye(xb.shape[1], dtype=np.float64)
    return np.linalg.solve(a, xb.T @ y)


def predict(w, x):
    xb = np.concatenate([x, np.ones((x.shape[0], 1), np.float32)], axis=1)
    return xb @ w


def main() -> int:
    groups: dict[str, list[Path]] = {}
    for p in sorted(DATA.glob("*-PSG.edf")):
        groups.setdefault(p.name[3:5], []).append(p)
    subjects = sorted(groups)
    print(f"{len(subjects)} subjects", flush=True)

    subj = {}
    for i, s in enumerate(subjects):
        ctxs, tgts = [], {h: [] for h in HORIZONS}
        for path in groups[s]:
            c, t = windows(load_raw_channel(path))
            ctxs.append(c)
            for h in HORIZONS:
                tgts[h].append(t[h])
        subj[s] = (np.concatenate(ctxs), {h: np.concatenate(tgts[h]) for h in HORIZONS})
        if i % 10 == 0:
            print(f"loaded {i + 1}/{len(subjects)}", flush=True)

    pack = json.load(open(PACK / "runs/isolated_residual_sleep/scores.json"))["folds"]

    result = {}
    for fi, held in enumerate(subjects):
        train_subs = [s for s in subjects if s != held]
        val_subs = train_subs[fi % (len(train_subs) - N_VAL_SUBJ):][:N_VAL_SUBJ]
        fit_subs = [s for s in train_subs if s not in val_subs]

        # train-fold-only scalar z-score, matching the pack's split_zscore
        cat = np.concatenate([subj[s][0].ravel() for s in train_subs]
                             + [subj[s][1][h] for s in train_subs for h in HORIZONS])
        mu, sd = float(cat.mean()), float(cat.std()) or 1.0

        def z(a):
            return ((a - mu) / sd).astype(np.float32)

        xfit = z(np.concatenate([subj[s][0] for s in fit_subs]))
        xval = z(np.concatenate([subj[s][0] for s in val_subs]))
        xall = z(np.concatenate([subj[s][0] for s in train_subs]))
        xte = z(subj[held][0])

        row = {}
        for h in HORIZONS:
            yfit = z(np.concatenate([subj[s][1][h] for s in fit_subs]))
            yval = z(np.concatenate([subj[s][1][h] for s in val_subs]))
            yall = z(np.concatenate([subj[s][1][h] for s in train_subs]))
            yte = z(subj[held][1][h])
            best_alpha, best_v = None, np.inf
            for a in ALPHAS:
                w = ridge_solve(xfit, yfit, a)
                v = float(np.mean((predict(w, xval) - yval) ** 2))
                if v < best_v:
                    best_alpha, best_v = a, v
            w = ridge_solve(xall, yall, best_alpha)
            mse = float(np.mean((predict(w, xte) - yte) ** 2))
            row[str(h)] = {"tuned_ridge": mse, "alpha": best_alpha,
                           "pack_ridge": pack[held][str(h)]["ridge"],
                           "pack_stack": pack[held][str(h)].get("ridge+osc+gru")}
        result[held] = row
        print(f"fold {held}: " + "  ".join(
            f"h{h} tuned {row[str(h)]['tuned_ridge']:.4f} vs stack {row[str(h)]['pack_stack']:.4f}"
            for h in HORIZONS), flush=True)

    OUT.write_text(json.dumps(result, indent=1))

    print("\n=== VERDICT ===")
    for h in map(str, HORIZONS):
        tuned = np.array([result[s][h]["tuned_ridge"] for s in subjects])
        stack = np.array([result[s][h]["pack_stack"] for s in subjects])
        weak = np.array([result[s][h]["pack_ridge"] for s in subjects])
        gap = tuned - stack
        rng = np.random.default_rng(0)
        boot = np.array([gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(10000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"h={h:>2}: weak-ridge {weak.mean():.4f}  tuned-ridge {tuned.mean():.4f}  "
              f"stack {stack.mean():.4f}  gap(tuned-stack) {gap.mean():+.4f} "
              f"[CI {lo:+.4f},{hi:+.4f}]  stack wins {int((gap > 0).sum())}/{len(gap)} folds",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
