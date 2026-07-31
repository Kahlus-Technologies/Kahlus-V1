"""BNCI2014_001 crossover sweep: does the Sleep-EDF short-horizon win replicate?

The paper is framed on BNCI2014_001, but the powered crossover result (F8) was
measured on Sleep-EDF. That is a framing gap, and the honest way to close it is
to run the *same* protocol on BNCI rather than to argue that it should transfer.

This script is `forecast_scaleup_paired.py` with the corpus swapped and nothing
else changed that matters:

  * subject-disjoint train/val/test split, same splitter, same seed policy
  * ridge alpha tuned per horizon on validation subjects
  * GRU (hidden, steps) tuned on the *same* validation subjects, several seeds
  * MSE scored per test subject, so the comparison is paired
  * strictly-future single-sample targets: the target is never inside the input

The symmetric tuning matters more here than anywhere else in the repo. An
earlier BNCI check (`bnci_isolated_forecast_check.py`) used a fixed ridge alpha
and one GRU seed, so "ridge wins on BNCI" from that run could not be
distinguished from "the GRU was under-tuned on BNCI". If ridge still wins under
a symmetric budget, the result is about the dataset, not the budget.

BNCI is 250 Hz against Sleep-EDF's 100 Hz, so horizons are chosen in
milliseconds and converted to samples per dataset. Comparing h=1 across the two
would otherwise compare 4 ms against 10 ms and call it replication.

A note on cost, because the first attempt at this run was misread as a crash.
The GRU grid is 4 configs x 3 seeds for tuning plus 3 seeds for the final fit,
and the largest config (hidden=64, steps=900) takes ~92 s on 3200 training rows.
That is ~10 min per fold, ~94 min per horizon, and ~14 h for all 9 horizons. The
first run therefore sat silent for 4 minutes and was recorded as "exited 0 with
no output" when in fact it had not yet reached its first print statement, and
the process was later reaped. Two things made a slow run indistinguishable from
a dead one, and both are fixed here:

  * the results file was written once, after all 81 fold-horizons, so an
    interrupted run left nothing behind no matter how much work it had done.
    The artifact is now rewritten after every horizon, and carries a `complete`
    flag plus the list of horizons actually finished, so a partial file is
    obviously partial rather than quietly short.
  * the only progress print was per horizon, i.e. every ~94 minutes. Each fold
    now reports as it lands, with a running ETA.

`--resume` reloads a partial artifact and skips the horizons already in it, so
this can be run in chunks without repeating work.

  PYTHONPATH=src python3 scripts/bnci_crossover_sweep.py --out runs/bnci_crossover
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from amrith_isolated_forecast_check import (  # noqa: E402
    L,
    _mse,
    gru_direct,
    ridge_fit,
    ridge_predict,
)
from forecast_scaleup_paired import (  # noqa: E402
    GRU_CONFIGS,
    GRU_SEEDS,
    RIDGE_ALPHAS,
    paired_stats,
    tune_ridge,
)
from analysis.build_bnci_ridge_tensors import (  # noqa: E402
    DEFAULT_CACHE,
    EEG_CHANNEL_NAMES,
    load_subject_eeg,
)

SEED = 0

#: BNCI's alpha grid has to run lower than Sleep-EDF's. The first pass selected
#: 0.1 -- the bottom of the inherited grid -- at every horizon, which means the
#: grid, not validation, was choosing the baseline's regularisation. A baseline
#: pinned against its own grid edge is exactly the under-tuning failure this
#: repo has already been caught by twice, so the grid is extended downward until
#: the selection is interior and the run asserts that it is.
BNCI_RIDGE_ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2) + tuple(RIDGE_ALPHAS)

#: Horizons in milliseconds, matched to the Sleep-EDF sweep so the two datasets
#: are compared at the same physical lead time rather than the same sample count.
HORIZONS_MS = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0)

#: C3: the sensorimotor channel this dataset is built around, and the channel the
#: earlier BNCI check used. Keeping it makes the two runs comparable.
CHANNEL = "C3"


def horizons_samples(fs: float) -> list[int]:
    return [max(1, int(round(ms / 1000.0 * fs))) for ms in HORIZONS_MS]


def make_windows(sig: np.ndarray, hs: list[int], *, stride: int,
                 max_windows: int, rng):
    """Context of L samples -> the single sample at i+L+h-1, strictly future."""
    need = L + max(hs)
    starts = np.arange(0, len(sig) - need, stride)
    if len(starts) == 0:
        return None
    if len(starts) > max_windows:
        starts = rng.choice(starts, size=max_windows, replace=False)
    ctx = np.stack([sig[i:i + L] for i in starts]).astype(np.float32)
    tgt = {h: np.array([sig[i + L + h - 1] for i in starts], dtype=np.float32)
           for h in hs}
    return ctx, tgt


def load_corpus(cache: Path, *, n_subjects: int, stride: int, max_windows: int):
    """One trace per subject, z-scored per subject.

    Per-subject z-scoring is what keeps a subject-held-out split honest: scaling
    by corpus statistics would leak test-subject variance into training.
    """
    ch = EEG_CHANNEL_NAMES.index(CHANNEL)
    out: dict[str, tuple] = {}
    fs_seen = None
    for s in range(1, n_subjects + 1):
        sig2d, fs = load_subject_eeg(cache, s)
        fs_seen = fs
        sig = sig2d[:, ch].astype(np.float64)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        hs = horizons_samples(fs)
        win = make_windows(sig, hs, stride=stride, max_windows=max_windows,
                           rng=np.random.default_rng(s))
        if win is None:
            print(f"  subject {s}: too short, skipped", flush=True)
            continue
        out[f"A{s:02d}"] = win
        print(f"  subject A{s:02d}: {win[0].shape[0]} windows", flush=True)
    return out, float(fs_seen or 250.0)


def tune_gru_seeded(train_ctx, train_y, val_ctx, val_y) -> tuple[int, int]:
    """Pick (hidden, steps) on validation, averaging over seeds.

    Averaging matters at this corpus size: with 9 subjects a single seed can
    select a configuration that only worked once, and the resulting model would
    then be judged on test as though that were its typical behaviour.
    """
    best = (np.inf, GRU_CONFIGS[0])
    for hidden, steps in GRU_CONFIGS:
        mses = [_mse(val_y, gru_direct(train_ctx, train_y, val_ctx,
                                       hidden=hidden, steps=steps, seed=sd))
                for sd in GRU_SEEDS]
        m = float(np.mean(mses))
        if m < best[0]:
            best = (m, (hidden, steps))
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=Path("runs/bnci_crossover"))
    ap.add_argument("--subjects", type=int, default=9)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--horizons-ms", type=float, nargs="*", default=None,
                    help="subset of HORIZONS_MS to run; default is all nine")
    ap.add_argument("--resume", action="store_true",
                    help="reload a partial artifact and skip finished horizons")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"loading BNCI2014_001 from {args.cache_dir}", flush=True)
    corpus, fs = load_corpus(args.cache_dir, n_subjects=args.subjects,
                             stride=args.stride, max_windows=args.max_windows)
    hs = horizons_samples(fs)
    print(f"fs={fs:g} Hz, horizons(samples)={hs}", flush=True)

    # BNCI has 9 subjects against Sleep-EDF's 78. A single 35/15/50 split leaves
    # 3 test subjects, and a 3-pair Wilcoxon cannot return p < 0.25 no matter how
    # large the effect -- the test would be unable to detect the very thing the
    # run exists to measure. Rotating every subject through the test position
    # instead yields 9 paired differences, each from a model that never saw that
    # subject. Tuning stays inside the fold: the validation subject is drawn from
    # the fold's training pool, so no fold ever tunes on its own test subject.
    subjects = sorted(corpus)
    folds = []
    for i, held in enumerate(subjects):
        pool = [s for s in subjects if s != held]
        val_subj = pool[i % len(pool)]
        folds.append({"test": [held],
                      "val": [val_subj],
                      "train": [s for s in pool if s != val_subj]})
    print(f"{len(folds)} leave-one-subject-out folds over {subjects}", flush=True)

    def stack(subjects, h=None):
        ctx = np.concatenate([corpus[s][0] for s in subjects])
        if h is None:
            return ctx
        return ctx, np.concatenate([corpus[s][1][h] for s in subjects])

    comparisons: dict[str, dict] = {}
    per_subject: dict[str, dict[str, dict[str, float]]] = {
        m: {} for m in ("kahlus_gru", "ridge_ar", "persistence", "mean_of_trace")
    }
    chosen: dict[str, dict] = {}
    alpha_edge_hits: list[str] = []

    path = args.out / "bnci_crossover_results.json"

    # Horizon selection, and resume. Both exist so that a 14-hour run can be
    # done in pieces without either repeating finished work or pretending an
    # unfinished run is a finished one.
    wanted = list(zip(hs, HORIZONS_MS))
    if args.horizons_ms is not None:
        keep = set(args.horizons_ms)
        wanted = [(h, ms) for h, ms in wanted if ms in keep]
        missing = keep - {ms for _, ms in zip(hs, HORIZONS_MS)}
        if missing:
            raise SystemExit(f"--horizons-ms values not in the grid: {sorted(missing)}")
    # What this invocation set out to measure, fixed before any resume filtering
    # so that "planned" keeps meaning the goal rather than the remaining work.
    planned_h = [h for h, _ in wanted]
    if args.resume and path.exists():
        prior = json.loads(path.read_text())
        comparisons = prior.get("comparisons", {})
        per_subject = prior.get("per_subject_mse", per_subject)
        chosen = prior.get("chosen", {})
        alpha_edge_hits = prior.get("ridge_alpha_grid_edge_selections", [])
        done = set(comparisons)
        wanted = [(h, ms) for h, ms in wanted if str(h) not in done]
        print(f"resuming: {len(done)} horizons already in {path}, "
              f"{len(wanted)} to go", flush=True)

    def snapshot(complete: bool) -> dict:
        """Build the artifact from whatever is finished so far.

        Written after every horizon rather than once at the end. The earlier
        version wrote only after all 81 fold-horizons, so an interrupted run
        left an empty directory that was indistinguishable from a crash --
        which is exactly how the first attempt at this run was misdiagnosed.
        """
        done_h = [h for h in hs if str(h) in comparisons]
        gru = [comparisons[str(h)]["mean_mse"]["kahlus_gru"] for h in done_h]
        ridge = [comparisons[str(h)]["mean_mse"]["ridge_ar"] for h in done_h]
        wins = [i for i in range(len(done_h)) if gru[i] < ridge[i]]
        ms_of = dict(zip(hs, HORIZONS_MS))
        # The crossover is a claim about the *shape* of the curve: the GRU wins
        # at the shortest horizon and loses at the long ones. A subset run
        # cannot answer that no matter how many folds it finished, so the flag
        # stays None unless the full nine-horizon sweep is complete. Reporting
        # True off two horizons would be answering a question nobody asked.
        full_sweep = complete and len(done_h) == len(hs)
        return {
            "schema": "bnci_crossover_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "BNCI2014_001 (BCI Competition IV-2a) via MOABB local cache",
            "source_dir": str(args.cache_dir),
            "channel": CHANNEL,
            "sampling_rate_hz": fs,
            "context_samples": L,
            "complete": complete,
            "full_sweep": full_sweep,
            # What this run set out to do, not what the module could do: a
            # subset run that finished must not look like a truncated full one.
            "horizons_planned": planned_h,
            "horizons_planned_ms": [ms_of[h] for h in planned_h],
            "horizons": done_h,
            "horizons_ms": [ms_of[h] for h in done_h],
            "ridge_alpha_grid": list(BNCI_RIDGE_ALPHAS),
            "ridge_alpha_grid_edge_selections": alpha_edge_hits,
            "gru_config_grid": [list(c) for c in GRU_CONFIGS],
            "gru_seeds": list(GRU_SEEDS),
            "chosen": chosen,
            "evaluation": (f"leave-one-subject-out; {len(folds)} folds, "
                           "tuning inside each fold"),
            "folds": folds,
            "comparisons": comparisons,
            "per_subject_mse": per_subject,
            "gru_wins_at_horizons_ms": [ms_of[done_h[i]] for i in wins],
            "replicates_sleepedf_crossover": (
                bool(wins and wins[0] == 0) if full_sweep else None),
        }

    total_folds = len(wanted) * len(folds)
    done_folds = 0
    t_start = time.time()

    for h, ms in wanted:
        subj_mse: dict[str, dict[str, float]] = {}
        fold_choices: list[dict] = []
        for fold in folds:
            t_fold = time.time()
            tr_ctx, tr_y = stack(fold["train"], h)
            va_ctx, va_y = stack(fold["val"], h)

            alpha, _ = tune_ridge(tr_ctx, tr_y, va_ctx, va_y,
                                  alphas=BNCI_RIDGE_ALPHAS)
            hidden, steps = tune_gru_seeded(tr_ctx, tr_y, va_ctx, va_y)
            if alpha in (BNCI_RIDGE_ALPHAS[0], BNCI_RIDGE_ALPHAS[-1]):
                alpha_edge_hits.append(f"h={h} fold={fold['test'][0]} alpha={alpha:g}")
            fold_choices.append({"test_subject": fold["test"][0],
                                 "val_subject": fold["val"][0],
                                 "ridge_alpha": alpha, "gru_hidden": hidden,
                                 "gru_steps": steps})

            w = ridge_fit(tr_ctx, tr_y, alpha)
            # Seeds are pooled by averaging predictions, matching F8: a single
            # seed's margin cannot then be mistaken for the model's margin.
            s = fold["test"][0]
            ctx, y = corpus[s][0], corpus[s][1][h]
            preds = np.mean([gru_direct(tr_ctx, tr_y, ctx, hidden=hidden,
                                        steps=steps, seed=sd)
                             for sd in GRU_SEEDS], axis=0)
            subj_mse[s] = {
                "kahlus_gru": _mse(y, preds),
                "ridge_ar": _mse(y, ridge_predict(w, ctx)),
                "persistence": _mse(y, ctx[:, -1]),
                "mean_of_trace": _mse(y, ctx.mean(axis=1)),
            }
            # A fold takes ~10 minutes. Reporting only per horizon meant ~94
            # minutes of silence, during which a live run and a dead one look
            # identical from outside.
            done_folds += 1
            eta = (time.time() - t_start) / done_folds * (total_folds - done_folds)
            print(f"  h={h:>3} ({ms:>6.0f} ms) fold {fold['test'][0]}  "
                  f"gru {subj_mse[s]['kahlus_gru']:.4f}  "
                  f"ridge {subj_mse[s]['ridge_ar']:.4f}  "
                  f"alpha={alpha:g} gru=({hidden},{steps})  "
                  f"[{done_folds}/{total_folds}, {time.time()-t_fold:.0f}s, "
                  f"eta {eta/60:.0f} min]", flush=True)
        chosen[str(h)] = {"horizon_ms": ms, "folds": fold_choices}
        test = subjects
        for m in per_subject:
            per_subject[m][str(h)] = {s: subj_mse[s][m] for s in test}

        mean_mse = {m: float(np.mean([subj_mse[s][m] for s in test]))
                    for m in subj_mse[test[0]]}
        delta = np.array([subj_mse[s]["ridge_ar"] - subj_mse[s]["kahlus_gru"]
                          for s in test])
        stats = paired_stats(delta)
        comparisons[str(h)] = {
            "horizon_ms": ms,
            "mean_mse": mean_mse,
            "gru_vs_ridge": {
                **stats,
                "n_subjects": len(test),
                "n_subjects_favoring_reference": int((delta > 0).sum()),
            },
        }
        print(f"h={h:>3} ({ms:>6.0f} ms)  gru {mean_mse['kahlus_gru']:.4f}  "
              f"ridge {mean_mse['ridge_ar']:.4f}  "
              f"persist {mean_mse['persistence']:.4f}  "
              f"favouring gru {int((delta > 0).sum())}/{len(test)}", flush=True)
        path.write_text(json.dumps(snapshot(complete=False), indent=1) + "\n")

    # Complete means this invocation finished its own plan; a deliberate
    # subset run is not a truncated one. full_sweep, computed inside the
    # snapshot, is the separate question of whether all nine are present.
    result = snapshot(complete=all(str(h) in comparisons for h in planned_h))
    gru_wins = result["gru_wins_at_horizons_ms"]
    if alpha_edge_hits:
        print(f"\nWARNING ridge alpha at a grid edge in {len(alpha_edge_hits)} "
              f"fold-horizons -- baseline may be under-tuned:\n  "
              + "\n  ".join(alpha_edge_hits[:10]), flush=True)
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwrote {path} (complete={result['complete']}, "
          f"{len(result['horizons'])}/{len(planned_h)} planned, "
          f"{len(result['horizons'])}/{len(hs)} of the full grid)")
    print(f"GRU wins at: {gru_wins} ms")
    print(f"replicates Sleep-EDF crossover: "
          f"{result['replicates_sleepedf_crossover']}")


if __name__ == "__main__":
    main()
