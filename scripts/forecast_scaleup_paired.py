"""Powered replication of the isolated EEG forecast check: all Sleep-EDF subjects,
per-subject scoring, paired statistics.

`amrith_isolated_forecast_check.py` established that the historical 126/127
overlap was the result, and that on a fair isolated forecast the GRU edges a
ridge-AR. That run had **two held-out test subjects and one seed**, so its
margins carried no error bars and no p-value. This script closes that gap:

  * all 78 sleep-cassette subjects, shuffled subject-held-out split
  * MSE computed **per test subject**, so model differences are paired
  * ridge alpha tuned per horizon on validation subjects carved out of train --
    the linear baseline has to be *well tuned* for a "we beat linear" claim to
    mean anything (Wu et al. 2025 name weak linear baselines as a field-wide gap)
  * several GRU seeds, so a margin cannot be seed luck
  * Wilcoxon signed-rank + bootstrap CI on the per-subject deltas
  * horizons extended to 2.56 s to locate where the advantage dies

Reuses the estimators and the random-walk/AR self-test from the original script;
the point here is statistical power, not new modelling.

  PYTHONPATH=src python3 scripts/forecast_scaleup_paired.py \
      --data /Users/aayu/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette \
      --out runs/forecast_scaleup
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amrith_isolated_forecast_check import (  # noqa: E402
    L,
    _mse,
    gru_direct,
    load_edf_channel,
    ridge_fit,
    ridge_predict,
    selftest,
)

#: 10 ms .. 2.56 s at 100 Hz. The tail is included to find where skill vanishes,
#: not because a win is expected there.
HORIZONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
#: Swept per horizon on held-out validation subjects. The grid runs high because
#: long horizons want heavy regularisation -- an earlier 24-subject pass selected
#: the top of a grid that stopped at 500, i.e. the baseline was still truncated.
RIDGE_ALPHAS = (0.1, 0.5, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0,
                2e5, 1e6, 5e6, 2e7)
#: (hidden, steps), swept on the same validation subjects as the ridge alpha.
#: Symmetric tuning budget is the point: "a tuned linear model matches the GRU"
#: is only honest if the GRU was tuned too.
GRU_CONFIGS = ((32, 300), (64, 300), (32, 900), (64, 900))
GRU_SEEDS = (0, 1, 2)
SEED = 0


def make_windows(sig: np.ndarray, *, stride: int, max_windows: int, rng):
    """Cut a trace into (context(L), {h: strictly-future target}).

    Target for horizon h is the single sample at index ``i + L + h - 1``: never
    inside ``sig[i:i+L]``. This is the whole point -- the historical bug scored a
    target window overlapping its own input.
    """
    need = L + max(HORIZONS)
    starts = np.arange(0, len(sig) - need, stride)
    if len(starts) == 0:
        return None
    if len(starts) > max_windows:
        starts = rng.choice(starts, size=max_windows, replace=False)
    ctx = np.stack([sig[i:i + L] for i in starts]).astype(np.float32)
    tgt = {h: np.array([sig[i + L + h - 1] for i in starts], dtype=np.float32)
           for h in HORIZONS}
    assert ctx.shape[1] == L
    return ctx, tgt


def _load_one(args: tuple[str, str, int, int, int]):
    """Worker: one recording -> (subject, ctx, {h: tgt}). None if unusable."""
    path, subject, stride, max_windows, seed = args
    sig = load_edf_channel(Path(path))
    if sig is None:
        return None
    win = make_windows(sig, stride=stride, max_windows=max_windows,
                       rng=np.random.default_rng(seed))
    if win is None:
        return None
    return subject, win[0], win[1]


def load_corpus(data: Path, *, n_subjects: int, stride: int, max_windows: int,
                workers: int):
    """Load every recording in parallel, grouped by subject."""
    psg = sorted(data.glob("*-PSG.edf"))
    by_subject: dict[str, list[Path]] = {}
    for p in psg:
        # SC4001E0 -> "SC400": both nights of one person share a subject id.
        by_subject.setdefault(p.name[:5], []).append(p)
    subjects = sorted(by_subject)[:n_subjects]

    jobs = [(str(p), s, stride, max_windows, i)
            for i, s in enumerate(subjects) for p in by_subject[s]]
    print(f"{len(subjects)} subjects, {len(jobs)} recordings, {workers} workers", flush=True)

    per_subject: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_load_one, jobs), 1):
            if res is None:
                continue
            subject, ctx, tgt = res
            per_subject.setdefault(subject, []).append((ctx, tgt))
            if i % 20 == 0:
                print(f"  loaded {i}/{len(jobs)}", flush=True)

    out = {}
    for subject, parts in per_subject.items():
        ctx = np.concatenate([c for c, _ in parts])
        tgt = {h: np.concatenate([t[h] for _, t in parts]) for h in HORIZONS}
        out[subject] = (ctx, tgt)
    return out


def split_subjects(subjects, seed: int) -> tuple[list, list, list]:
    """Deterministic subject-disjoint (train, val, test) partition.

    Shared with `validate_forecast_claim.py` on purpose: if the validation suite
    re-derived the split it could silently drift and end up validating a
    different partition than the one the finding reports.
    """
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(sorted(subjects)))
    n_test = int(round(0.35 * len(order)))
    n_val = int(round(0.15 * len(order)))
    test, val = sorted(order[:n_test]), sorted(order[n_test:n_test + n_val])
    train = sorted(order[n_test + n_val:])
    # Disjointness is the one property the whole result rests on; assert it
    # pairwise (a 3-way intersection can be empty while two sets still overlap).
    assert not (set(test) & set(train)), "train/test subject overlap"
    assert not (set(test) & set(val)), "val/test subject overlap"
    assert not (set(train) & set(val)), "train/val subject overlap"
    return train, val, test


def tune_ridge(train_ctx, train_y, val_ctx, val_y, *, alphas=None) -> tuple[float, np.ndarray]:
    """Pick alpha on validation subjects. A linear baseline is only a fair
    comparator if it got its own hyperparameter search.

    `alphas` overrides the default grid. The default is left alone so the F8 and
    F9 numbers stay reproducible; the capacity experiment passes a wider grid
    because an edge selection there under-tunes the baseline and would inflate
    the neural advantage it is trying to measure.
    """
    RIDGE_ALPHAS_ = tuple(alphas) if alphas is not None else RIDGE_ALPHAS
    best = (np.inf, RIDGE_ALPHAS_[0], None)
    for alpha in RIDGE_ALPHAS_:
        w = ridge_fit(train_ctx, train_y, alpha)
        mse = _mse(val_y, ridge_predict(w, val_ctx))
        if mse < best[0]:
            best = (mse, alpha, w)
    if best[1] in (RIDGE_ALPHAS_[0], RIDGE_ALPHAS_[-1]):
        print(f"    WARNING ridge alpha {best[1]:g} is at a grid edge -- "
              f"baseline may still be under-tuned", flush=True)
    return best[1], best[2]


def tune_gru(train_ctx, train_y, val_ctx, val_y) -> tuple[int, int]:
    """Pick (hidden, steps) on the same validation subjects the ridge used."""
    best = (np.inf, GRU_CONFIGS[0])
    for hidden, steps in GRU_CONFIGS:
        pred = gru_direct(train_ctx, train_y, val_ctx, hidden=hidden, steps=steps)
        mse = _mse(val_y, pred)
        if mse < best[0]:
            best = (mse, (hidden, steps))
    return best[1]


def paired_stats(delta: np.ndarray, *, n_boot: int = 10_000, seed: int = SEED) -> dict:
    """Per-subject paired summary for ``delta = other - reference`` (MSE units).

    Positive mean_delta => the reference model has lower error. Wilcoxon is the
    headline test (n~30, no normality assumption); the bootstrap CI carries the
    effect size.
    """
    from scipy.stats import wilcoxon

    delta = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(delta, size=(n_boot, delta.size), replace=True).mean(axis=1)
    try:
        stat, p = wilcoxon(delta)
        p = float(p)
    except ValueError:  # all-zero differences
        p = 1.0
    return {
        "n_subjects": int(delta.size),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "wilcoxon_p": p,
        "n_subjects_favoring_reference": int((delta > 0).sum()),
    }


def evaluate(corpus: dict, train_subj, val_subj, test_subj) -> dict:
    """Fit every model per horizon; score each test subject separately."""
    def pool(subjects):
        ctx = np.concatenate([corpus[s][0] for s in subjects])
        tgt = {h: np.concatenate([corpus[s][1][h] for s in subjects]) for h in HORIZONS}
        return ctx, tgt

    tr_ctx, tr_tgt = pool(train_subj)
    va_ctx, va_tgt = pool(val_subj)
    # One concatenated test matrix so each GRU is trained once per (horizon, seed)
    # and predicts every subject in a single forward pass; `bounds` splits it back.
    te_ctx = np.concatenate([corpus[s][0] for s in test_subj])
    edges = np.cumsum([0] + [corpus[s][0].shape[0] for s in test_subj])
    bounds = {s: (int(edges[i]), int(edges[i + 1])) for i, s in enumerate(test_subj)}
    print(f"train windows {tr_ctx.shape[0]}  val {va_ctx.shape[0]}  "
          f"test {te_ctx.shape[0]} over {len(test_subj)} subjects", flush=True)

    methods = ("mean_of_trace", "persistence", "ridge_ar", "kahlus_gru")
    # per_subject[method][h] -> {subject: mse}
    per_subject = {m: {h: {} for h in HORIZONS} for m in methods}
    chosen_alpha, chosen_gru, seed_spread = {}, {}, {}

    for h in HORIZONS:
        alpha, w = tune_ridge(tr_ctx, tr_tgt[h], va_ctx, va_tgt[h])
        chosen_alpha[h] = alpha
        hidden, steps = tune_gru(tr_ctx, tr_tgt[h], va_ctx, va_tgt[h])
        chosen_gru[h] = {"hidden": hidden, "steps": steps}

        gru_preds = {seed: gru_direct(tr_ctx, tr_tgt[h], te_ctx,
                                      hidden=hidden, steps=steps, seed=seed)
                     for seed in GRU_SEEDS}
        seed_spread[h] = [
            float(np.mean([_mse(corpus[s][1][h], gru_preds[seed][slice(*bounds[s])])
                           for s in test_subj]))
            for seed in GRU_SEEDS
        ]

        for s in test_subj:
            ctx, y = corpus[s][0], corpus[s][1][h]
            per_subject["mean_of_trace"][h][s] = _mse(y, ctx.mean(axis=1))
            per_subject["persistence"][h][s] = _mse(y, ctx[:, -1])
            per_subject["ridge_ar"][h][s] = _mse(y, ridge_predict(w, ctx))
            # Average the seeds' predictions' errors, not the predictions: the
            # claim is about the method, not about an ensemble we never proposed.
            per_subject["kahlus_gru"][h][s] = float(np.mean(
                [_mse(y, gru_preds[seed][slice(*bounds[s])]) for seed in GRU_SEEDS]))

        row = {m: float(np.mean(list(per_subject[m][h].values()))) for m in methods}
        print(f"  h={h:>3} alpha={alpha:<8g} gru={hidden}x{steps:<4} "
              f"ridge {row['ridge_ar']:.4f}  gru {row['kahlus_gru']:.4f}  "
              f"persist {row['persistence']:.4f}  "
              f"seed_spread {np.std(seed_spread[h]):.4f}", flush=True)

    return {"per_subject": per_subject, "alpha": chosen_alpha,
            "gru_config": chosen_gru, "gru_seed_pooled_mse": seed_spread,
            "methods": methods}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/forecast_scaleup"))
    ap.add_argument("--subjects", type=int, default=78)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ridge-only", action="store_true",
                    help="sweep the ridge alpha grid and exit; used to confirm the "
                         "linear baseline is not pinned at a grid edge")
    args = ap.parse_args()

    # Same gate the original run used, plus a check on the new paired-stats path.
    selftest()
    _d = np.array([0.05] * 12)
    _s = paired_stats(_d, n_boot=2000)
    assert _s["wilcoxon_p"] < 0.01 and _s["ci95_low"] > 0, f"paired_stats broken: {_s}"
    _s0 = paired_stats(np.array([0.05, -0.05] * 8), n_boot=2000)
    assert _s0["wilcoxon_p"] > 0.5, f"paired_stats must not fire on a null: {_s0}"
    print("selftest OK (estimators + paired stats)", flush=True)
    if args.selftest:
        return 0
    if args.data is None:
        ap.error("--data required (or use --selftest)")

    corpus = load_corpus(args.data, n_subjects=args.subjects, stride=args.stride,
                         max_windows=args.max_windows, workers=args.workers)
    subjects = sorted(corpus)
    if len(subjects) < 12:
        ap.error(f"need >=12 subjects for a paired test, got {len(subjects)}")

    train_subj, val_subj, test_subj = split_subjects(subjects, SEED)
    if min(len(train_subj), len(val_subj), len(test_subj)) < 3:
        ap.error(f"split too small: train {len(train_subj)} val {len(val_subj)} "
                 f"test {len(test_subj)} -- need >=3 each, pass more --subjects")
    print(f"split: train {len(train_subj)}  val {len(val_subj)}  test {len(test_subj)}",
          flush=True)

    if args.ridge_only:
        tr_ctx = np.concatenate([corpus[s][0] for s in train_subj])
        va_ctx = np.concatenate([corpus[s][0] for s in val_subj])
        print(f"{'h':>4} {'alpha':>10} {'test_mse':>10}")
        for h in HORIZONS:
            tr_y = np.concatenate([corpus[s][1][h] for s in train_subj])
            va_y = np.concatenate([corpus[s][1][h] for s in val_subj])
            alpha, w = tune_ridge(tr_ctx, tr_y, va_ctx, va_y)
            test_mse = float(np.mean([
                _mse(corpus[s][1][h], ridge_predict(w, corpus[s][0])) for s in test_subj]))
            print(f"{h:>4} {alpha:>10g} {test_mse:>10.4f}", flush=True)
        return 0

    res = evaluate(corpus, train_subj, val_subj, test_subj)
    ps = res["per_subject"]

    comparisons = {}
    for h in HORIZONS:
        gru = np.array([ps["kahlus_gru"][h][s] for s in test_subj])
        comparisons[str(h)] = {
            "mean_mse": {m: float(np.mean([ps[m][h][s] for s in test_subj]))
                         for m in res["methods"]},
            "gru_vs_ridge": paired_stats(
                np.array([ps["ridge_ar"][h][s] for s in test_subj]) - gru),
            "gru_vs_persistence": paired_stats(
                np.array([ps["persistence"][h][s] for s in test_subj]) - gru),
        }

    payload = {
        "schema": "kahlus.forecast_scaleup.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "protocol": "isolated strictly-future single-sample forecast; zero input/target overlap",
        "context_samples": L,
        "sampling_rate_hz": 100,
        "horizons": list(HORIZONS),
        "gru_seeds": list(GRU_SEEDS),
        "ridge_alpha_grid": list(RIDGE_ALPHAS),
        "gru_config_grid": [list(c) for c in GRU_CONFIGS],
        "chosen_ridge_alpha": {str(k): v for k, v in res["alpha"].items()},
        "chosen_gru_config": {str(k): v for k, v in res["gru_config"].items()},
        "gru_seed_pooled_mse": {str(k): v for k, v in res["gru_seed_pooled_mse"].items()},
        "split": {"train": train_subj, "val": val_subj, "test": test_subj},
        "comparisons": comparisons,
        "per_subject_mse": {m: {str(h): ps[m][h] for h in HORIZONS} for m in res["methods"]},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== paired, per-subject (positive delta = GRU better) ===")
    print(f"{'h':>4} {'ridge':>8} {'gru':>8} {'Δ':>8} {'95% CI':>18} {'p':>10} {'win/n':>7}")
    for h in HORIZONS:
        c = comparisons[str(h)]
        v = c["gru_vs_ridge"]
        print(f"{h:>4} {c['mean_mse']['ridge_ar']:>8.4f} {c['mean_mse']['kahlus_gru']:>8.4f} "
              f"{v['mean_delta']:>8.4f} "
              f"[{v['ci95_low']:>7.4f},{v['ci95_high']:>7.4f}] {v['wilcoxon_p']:>10.2e} "
              f"{v['n_subjects_favoring_reference']:>3}/{v['n_subjects']:<3}")
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
