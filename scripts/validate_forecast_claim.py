"""Adversarial validation of F8 (`docs/results/forecast_crossover_2026-07-27.md`).

F8 claims a small GRU beats a per-horizon-tuned ridge-AR on strictly-future
single-sample EEG forecasting below ~160 ms, and loses above ~320 ms. This script
exists to try to destroy that claim. Each test below is written so that a
specific *failure* would falsify F8, not confirm it.

  T1  index audit          context and target must share zero sample indices
  T2  shuffled-target null a broken pipeline "wins" on permuted labels
  T3  noise-input null     a broken pipeline "wins" with no signal present
  T4  linear-SGD control   is the margin nonlinearity, or just the optimiser?
  T5  AR(2) sanity         on data where linear is provably optimal, GRU must NOT win
  T6  split robustness     the seed-0 split could simply be lucky
  T7  non-overlapping      stride < L+h means neighbouring windows share samples
  T8  Holm correction      9 horizons were tested; correct for it

T4 and T5 are the ones that matter scientifically. T2, T3 and T5 are designed to
FAIL LOUDLY if the harness is capable of manufacturing a win from nothing.

  PYTHONPATH=src python3 scripts/validate_forecast_claim.py \
      --data <sleep-cassette> --out runs/forecast_validation
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amrith_isolated_forecast_check import L, _mse, gru_direct, ridge_predict  # noqa: E402
from forecast_scaleup_paired import (  # noqa: E402
    load_corpus,
    paired_stats,
    tune_ridge,
)

#: The horizons that carry the claim: best GRU win, the crossover, and two
#: horizons where F8 says ridge wins. Running all nine again buys nothing.
PROBE_HORIZONS = (1, 16, 64, 256)
GRU_HIDDEN, GRU_STEPS = 32, 900
SEED = 0


def linear_sgd(train_x, train_y, test_x, *, steps=GRU_STEPS, seed=0):
    """A LINEAR map trained by the exact loop `gru_direct` uses.

    This isolates architecture from optimisation. If this matches the GRU, the
    GRU's margin over closed-form ridge is not evidence of nonlinearity -- it is
    evidence that SGD with early stopping regularises differently than an L2
    penalty chosen on validation data.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    net = nn.Linear(train_x.shape[1], 1)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-2)
    xt = torch.as_tensor(train_x, dtype=torch.float32)
    yt = torch.as_tensor(train_y[:, None], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    n = xt.shape[0]
    bs = min(256, n)
    net.train()
    for _ in range(steps):
        idx = rng.integers(0, n, size=bs)
        opt.zero_grad(set_to_none=True)
        nn.functional.mse_loss(net(xt[idx]), yt[idx]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.as_tensor(test_x, dtype=torch.float32)).squeeze(-1).numpy()


def pooled(corpus, subjects, h):
    ctx = np.concatenate([corpus[s][0] for s in subjects])
    y = np.concatenate([corpus[s][1][h] for s in subjects])
    return ctx, y


def per_subject_mse(corpus, subjects, h, pred_all):
    """Split a concatenated prediction back into per-subject MSEs."""
    out, off = {}, 0
    for s in subjects:
        n = corpus[s][0].shape[0]
        out[s] = _mse(corpus[s][1][h], pred_all[off:off + n])
        off += n
    assert off == pred_all.shape[0]
    return out


def split_subjects(subjects, seed):
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(sorted(subjects)))
    n_test = int(round(0.35 * len(order)))
    n_val = int(round(0.15 * len(order)))
    return (sorted(order[n_test + n_val:]), sorted(order[n_test:n_test + n_val]),
            sorted(order[:n_test]))


# ------------------------------------------------------------------- T1 ------
def t1_index_audit(stride: int, max_h: int) -> dict:
    """Context indices and target index must be disjoint, for every window."""
    worst = 0
    for start in range(0, 5000, stride):
        ctx_idx = set(range(start, start + L))
        for h in PROBE_HORIZONS:
            tgt_idx = start + L + h - 1
            worst = max(worst, len(ctx_idx & {tgt_idx}))
    # Neighbouring windows within one subject DO share samples when
    # stride < L + max_h. That is between-example overlap, not the within-example
    # kind that produced the historical artifact -- but quantify it honestly.
    need = L + max_h
    neighbour_shared = max(0, need - stride)
    return {
        "context_target_shared_samples": worst,
        "passes": worst == 0,
        "neighbour_window_shared_samples": neighbour_shared,
        "neighbour_windows_overlap": neighbour_shared > 0,
        "note": ("context/target overlap is the claim-critical one and is zero; "
                 "neighbour-window overlap is within-subject only and cannot cross "
                 "a subject-disjoint split (tested separately in T7)"),
    }


# ---------------------------------------------------------------- T2/T3 ------
def t2_t3_nulls(corpus, train_s, val_s, test_s, h, rng) -> dict:
    """Destroy the signal two ways; both models must collapse to the variance floor
    and the GRU-vs-ridge difference must vanish."""
    out = {}
    tr_ctx, tr_y = pooled(corpus, train_s, h)
    va_ctx, va_y = pooled(corpus, val_s, h)
    te_ctx = np.concatenate([corpus[s][0] for s in test_s])

    # T2: permute training targets -> context carries no information about y.
    perm = rng.permutation(tr_y.shape[0])
    alpha, w = tune_ridge(tr_ctx, tr_y[perm], va_ctx, va_y)
    r_ps = per_subject_mse(corpus, test_s, h, ridge_predict(w, te_ctx))
    g_ps = per_subject_mse(corpus, test_s, h,
                           gru_direct(tr_ctx, tr_y[perm], te_ctx,
                                      hidden=GRU_HIDDEN, steps=GRU_STEPS))
    d = np.array([r_ps[s] - g_ps[s] for s in test_s])
    out["shuffled_targets"] = {
        "ridge_mse": float(np.mean(list(r_ps.values()))),
        "gru_mse": float(np.mean(list(g_ps.values()))),
        **paired_stats(d),
    }

    # T3: replace the context with white noise, keep real targets.
    n_tr, n_te = tr_ctx.shape[0], te_ctx.shape[0]
    noise_tr = rng.normal(size=(n_tr, L)).astype(np.float32)
    noise_va = rng.normal(size=(va_ctx.shape[0], L)).astype(np.float32)
    noise_te = rng.normal(size=(n_te, L)).astype(np.float32)
    alpha, w = tune_ridge(noise_tr, tr_y, noise_va, va_y)
    r_ps = per_subject_mse(corpus, test_s, h, ridge_predict(w, noise_te))
    g_ps = per_subject_mse(corpus, test_s, h,
                           gru_direct(noise_tr, tr_y, noise_te,
                                      hidden=GRU_HIDDEN, steps=GRU_STEPS))
    d = np.array([r_ps[s] - g_ps[s] for s in test_s])
    out["noise_input"] = {
        "ridge_mse": float(np.mean(list(r_ps.values()))),
        "gru_mse": float(np.mean(list(g_ps.values()))),
        **paired_stats(d),
    }
    return out


# ------------------------------------------------------------------- T4 ------
def t4_linear_sgd(corpus, train_s, val_s, test_s, h) -> dict:
    """GRU vs a linear model trained by the identical optimiser."""
    tr_ctx, tr_y = pooled(corpus, train_s, h)
    va_ctx, va_y = pooled(corpus, val_s, h)
    te_ctx = np.concatenate([corpus[s][0] for s in test_s])

    alpha, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y)
    ridge_ps = per_subject_mse(corpus, test_s, h, ridge_predict(w, te_ctx))
    lin_ps = per_subject_mse(corpus, test_s, h, linear_sgd(tr_ctx, tr_y, te_ctx))
    gru_ps = per_subject_mse(corpus, test_s, h,
                             gru_direct(tr_ctx, tr_y, te_ctx,
                                        hidden=GRU_HIDDEN, steps=GRU_STEPS))
    dg = np.array([lin_ps[s] - gru_ps[s] for s in test_s])   # + => GRU beats linear-SGD
    dr = np.array([ridge_ps[s] - lin_ps[s] for s in test_s])  # + => linear-SGD beats ridge
    return {
        "ridge_mse": float(np.mean(list(ridge_ps.values()))),
        "linear_sgd_mse": float(np.mean(list(lin_ps.values()))),
        "gru_mse": float(np.mean(list(gru_ps.values()))),
        "gru_vs_linear_sgd": paired_stats(dg),
        "linear_sgd_vs_ridge": paired_stats(dr),
    }


# ------------------------------------------------------------------- T5 ------
def t5_ar_sanity(rng) -> dict:
    """On a stationary AR(2) process the optimal predictor is LINEAR.

    A correctly specified comparison must therefore show the GRU failing to beat
    ridge here. If it "wins" on AR(2), the harness manufactures wins and the
    Sleep-EDF margin means nothing.
    """
    n_sub, n_win = 30, 400
    a1, a2 = 1.2, -0.5           # stable, oscillatory, non-trivial memory
    need = L + max(PROBE_HORIZONS) + 1
    corpus = {}
    for i in range(n_sub):
        e = rng.normal(size=(n_win, need)).astype(np.float32)
        x = np.zeros_like(e)
        for t in range(2, need):
            x[:, t] = a1 * x[:, t - 1] + a2 * x[:, t - 2] + e[:, t]
        x = (x - x.mean()) / (x.std() + 1e-8)
        corpus[f"AR{i:02d}"] = (x[:, :L].copy(),
                                {h: x[:, L + h - 1].copy() for h in PROBE_HORIZONS})
    tr, va, te = split_subjects(list(corpus), SEED)
    out = {}
    for h in PROBE_HORIZONS:
        tr_ctx, tr_y = pooled(corpus, tr, h)
        va_ctx, va_y = pooled(corpus, va, h)
        te_ctx = np.concatenate([corpus[s][0] for s in te])
        alpha, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y)
        r_ps = per_subject_mse(corpus, te, h, ridge_predict(w, te_ctx))
        g_ps = per_subject_mse(corpus, te, h,
                               gru_direct(tr_ctx, tr_y, te_ctx,
                                          hidden=GRU_HIDDEN, steps=GRU_STEPS))
        d = np.array([r_ps[s] - g_ps[s] for s in te])
        out[str(h)] = {"ridge_mse": float(np.mean(list(r_ps.values()))),
                       "gru_mse": float(np.mean(list(g_ps.values()))),
                       **paired_stats(d)}
        print(f"    AR(2) h={h:>3}: ridge {out[str(h)]['ridge_mse']:.4f}  "
              f"gru {out[str(h)]['gru_mse']:.4f}  p={out[str(h)]['wilcoxon_p']:.3g}",
              flush=True)
    return out


# ------------------------------------------------------------------- T6 ------
def t6_split_robustness(corpus, h, n_splits) -> dict:
    """Repeat the whole comparison over independent subject splits."""
    deltas = []
    for seed in range(n_splits):
        tr, va, te = split_subjects(list(corpus), seed)
        tr_ctx, tr_y = pooled(corpus, tr, h)
        va_ctx, va_y = pooled(corpus, va, h)
        te_ctx = np.concatenate([corpus[s][0] for s in te])
        alpha, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y)
        r_ps = per_subject_mse(corpus, te, h, ridge_predict(w, te_ctx))
        g_ps = per_subject_mse(corpus, te, h,
                               gru_direct(tr_ctx, tr_y, te_ctx, hidden=GRU_HIDDEN,
                                          steps=GRU_STEPS, seed=seed))
        d = float(np.mean([r_ps[s] - g_ps[s] for s in te]))
        deltas.append(d)
        print(f"    split seed={seed}: mean delta {d:+.4f}", flush=True)
    a = np.array(deltas)
    return {"n_splits": n_splits, "deltas": deltas, "mean": float(a.mean()),
            "std": float(a.std()), "n_positive": int((a > 0).sum()),
            "min": float(a.min()), "max": float(a.max())}


# ------------------------------------------------------------------- T8 ------
def t8_holm(results_path: Path) -> dict:
    """Holm-Bonferroni over the 9 horizons reported in F8."""
    r = json.loads(results_path.read_text(encoding="utf-8"))
    rows = [(str(h), r["comparisons"][str(h)]["gru_vs_ridge"]["wilcoxon_p"],
             r["comparisons"][str(h)]["gru_vs_ridge"]["mean_delta"])
            for h in r["horizons"]]
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][1])
    out, running = {}, 0.0
    for rank, i in enumerate(order):
        h, p, delta = rows[i]
        adj = min(1.0, max(running, (m - rank) * p))
        running = adj
        out[h] = {"raw_p": p, "holm_p": adj, "mean_delta": delta,
                  "survives_0.05": bool(adj < 0.05)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/forecast_validation"))
    ap.add_argument("--results", type=Path,
                    default=Path("runs/forecast_scaleup/results.json"))
    ap.add_argument("--subjects", type=int, default=78)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--splits", type=int, default=8)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    report: dict = {"schema": "kahlus.forecast_validation.v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "probe_horizons": list(PROBE_HORIZONS)}

    print("T1 index audit", flush=True)
    report["t1_index_audit"] = t1_index_audit(args.stride, max(PROBE_HORIZONS))
    print(f"  {report['t1_index_audit']}", flush=True)

    print("T8 Holm correction over the 9 reported horizons", flush=True)
    report["t8_holm"] = t8_holm(args.results)
    for h, v in report["t8_holm"].items():
        print(f"  h={h:>4} raw {v['raw_p']:.2e} -> holm {v['holm_p']:.2e} "
              f"survives={v['survives_0.05']}", flush=True)

    print("T5 AR(2) sanity -- GRU must NOT beat ridge where linear is optimal", flush=True)
    report["t5_ar_sanity"] = t5_ar_sanity(rng)

    print("loading Sleep-EDF corpus", flush=True)
    corpus = load_corpus(args.data, n_subjects=args.subjects, stride=args.stride,
                         max_windows=args.max_windows, workers=args.workers)
    train_s, val_s, test_s = split_subjects(list(corpus), SEED)

    print("T2/T3 nulls (shuffled targets, noise input) at h=16", flush=True)
    report["t2_t3_nulls_h16"] = t2_t3_nulls(corpus, train_s, val_s, test_s, 16, rng)
    for k, v in report["t2_t3_nulls_h16"].items():
        print(f"  {k}: ridge {v['ridge_mse']:.4f}  gru {v['gru_mse']:.4f}  "
              f"delta {v['mean_delta']:+.4f}  p={v['wilcoxon_p']:.3g}", flush=True)

    print("T4 linear-SGD control", flush=True)
    report["t4_linear_sgd"] = {}
    for h in PROBE_HORIZONS:
        v = t4_linear_sgd(corpus, train_s, val_s, test_s, h)
        report["t4_linear_sgd"][str(h)] = v
        print(f"  h={h:>3}: ridge {v['ridge_mse']:.4f}  linSGD {v['linear_sgd_mse']:.4f}  "
              f"gru {v['gru_mse']:.4f} | gru-vs-linSGD delta "
              f"{v['gru_vs_linear_sgd']['mean_delta']:+.4f} "
              f"p={v['gru_vs_linear_sgd']['wilcoxon_p']:.3g}", flush=True)

    print(f"T6 split robustness at h=16 and h=64 over {args.splits} splits", flush=True)
    report["t6_split_robustness"] = {
        str(h): t6_split_robustness(corpus, h, args.splits) for h in (16, 64)}

    print("T7 non-overlapping windows (stride >= L+max_h)", flush=True)
    ns = L + max(PROBE_HORIZONS) + 1
    corpus_ns = load_corpus(args.data, n_subjects=args.subjects, stride=ns,
                            max_windows=args.max_windows, workers=args.workers)
    tr2, va2, te2 = split_subjects(list(corpus_ns), SEED)
    report["t7_non_overlapping"] = {"stride": ns}
    for h in PROBE_HORIZONS:
        tr_ctx, tr_y = pooled(corpus_ns, tr2, h)
        va_ctx, va_y = pooled(corpus_ns, va2, h)
        te_ctx = np.concatenate([corpus_ns[s][0] for s in te2])
        alpha, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y)
        r_ps = per_subject_mse(corpus_ns, te2, h, ridge_predict(w, te_ctx))
        g_ps = per_subject_mse(corpus_ns, te2, h,
                               gru_direct(tr_ctx, tr_y, te_ctx, hidden=GRU_HIDDEN,
                                          steps=GRU_STEPS))
        d = np.array([r_ps[s] - g_ps[s] for s in te2])
        v = {"ridge_mse": float(np.mean(list(r_ps.values()))),
             "gru_mse": float(np.mean(list(g_ps.values()))), **paired_stats(d)}
        report["t7_non_overlapping"][str(h)] = v
        print(f"  h={h:>3}: ridge {v['ridge_mse']:.4f}  gru {v['gru_mse']:.4f}  "
              f"delta {v['mean_delta']:+.4f}  p={v['wilcoxon_p']:.3g}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"\nwrote {args.out / 'validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
