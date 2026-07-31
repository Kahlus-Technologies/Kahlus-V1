"""F12: can sleep stage be forecast 60 s before it happens, from EEG alone?

This is H2 as it was actually written -- "cortical state transitions are
predictable at >=30 s lead from EEG alone" -- rather than a proxy for it. F11
forecast the *amplitude* of a band envelope, which is noisy and which no
clinician has ever wanted to know. The state is the thing worth predicting.

WHAT MAKES THIS HARD, AND WHY THE BASELINE IS THE WHOLE EXPERIMENT.

Sleep stages last many minutes, so "whatever stage it is now, it will still be
that in 60 s" is right most of the time. Any model that reports raw accuracy
against a majority-class baseline is measuring stage duration, not prediction.
So persistence is the baseline that matters, and it is a brutal one.

The analysis that decides the question is therefore the TRANSITION subset:
windows where the stage at t+h differs from the stage at t. There, persistence
is wrong by construction, and beating it means the model saw the change coming.
Everything else is state tracking.

DESIGN.
  inputs   multi-band causal envelopes (delta..low gamma) from BOTH EEG
           channels, decimated to 10 Hz, over a 12.8 s context window
  target   scored sleep stage at t+h, five classes (W/N1/N2/N3/REM)
  horizons 10, 30, 60 s
  model    multinomial logistic regression, L2 strength selected on held-out
           selection subjects, never on the test subjects
  splits   subject-disjoint fit/select/test
  guard    the same strictly-future assertion used everywhere in this repo

FOUR INPUT SETS, because "beats persistence" and "adds to persistence" are
different questions and only the second one is worth asking.

  stage_only     one-hot of the CURRENT stage and nothing else. This is
                 persistence expressed as a model, and it is the reference.
  eeg_only       EEG envelopes, no stage. Answers "from EEG alone" as H2 words
                 it -- but a first run showed it LOSES to persistence overall
                 (0.599 vs 0.734 balanced accuracy at 60 s), because knowing the
                 current stage is worth more than any amount of EEG when stages
                 last minutes. That is the honest headline for this input set.
  eeg_plus_stage EEG envelopes AND the current stage. This is the question that
                 matters: given that you already know where the sleeper is, does
                 the EEG tell you anything about where they are going? A real
                 system has the current stage for free.
  all_plus_stage adds EOG and EMG. Clinically achievable on a real
                 polysomnograph, but NOT evidence about EEG -- and during sleep
                 EOG and EMG are part of how REM is *defined*, so they carry
                 state information by construction rather than by prediction.

METRICS. Balanced accuracy and macro F1, never raw accuracy: the classes are
badly imbalanced and raw accuracy would reward predicting N2 forever. Reported
against persistence and against the majority class, on all windows and on the
transition subset separately.

PART TWO: TRANSITION DETECTION, which is the question the 5-class task cannot
answer. Transitions are 4-19% of windows, so a classifier minimising loss over
all windows correctly learns to persist and is never asked to spot change --
which is exactly what the numbers above show. Recast as a binary question,
"will the stage change within the next h seconds", the model is finally being
scored on the thing H2 cares about.

Scored with ROC AUC and average precision rather than accuracy, because the
positive class is rare and thresholded accuracy would be dominated by the base
rate. The reference is stage_only: transition probability genuinely depends on
the current stage (N1 is unstable, N3 is not), so "does EEG beat chance" is too
easy a bar and "does EEG add over the current stage" is the honest one.

PART THREE: THE LABEL-NOISE CEILING, which decides how to read any of the above.

A model cannot predict a label more reliably than the label is defined. Manual
sleep staging has a meta-analytic inter-scorer Cohen's kappa of 0.76 overall,
and per stage: W 0.70, N1 0.24, N2 0.57, N3 0.57, REM 0.69 (Lee et al., J Clin
Sleep Med 2022; corroborated by Danker-Hopfe et al., J Sleep Res 2009, which
reports 82.0% raw epoch agreement under AASM).

N1's kappa of 0.24 is barely above chance -- experts essentially do not agree on
it. And a transition is defined by TWO labels, so its reliability is worse than
either. If two scorers agree ~82% per epoch, both endpoints being reproducible
is roughly 0.82^2 ~ 0.67, meaning about a third of scored transitions may be
disagreement rather than physiology.

So the analysis stratifies transitions by whether either endpoint is N1. If AUC
rises materially once N1 transitions are excluded, the bottleneck is the label,
not the model -- and the honest headline is the reliable-subset number with the
ceiling stated, not a single pooled figure that silently averages in coin flips.

  PYTHONPATH=src python3 scripts/stage_forecast_sleepedf.py --out runs/stage_forecast
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envelope_horizon_sleepedf import (  # noqa: E402
    BANDS,
    DECIM,
    ENV_FS,
    FS,
    L,
    SEED,
    causal_envelope,
    load_channels,
    sleep_span,
    zscore,
)
from envelope_stage_decomposition import stage_series  # noqa: E402

HORIZONS = (100, 300, 600)              # 10, 30, 60 s at ENV_FS
#: 32 at 10 Hz is one window per 3.2 s. Stride 16 oversampled: adjacent windows
#: share 95%+ of their context, so it doubled memory for near-duplicate rows.
STRIDE = 32
CLASSES = ("W", "N1", "N2", "N3", "REM")
#: Inverse L2 strength for LogisticRegression. Extended down to 1e-7 and up to
#: 1e2 after the first run selected the bottom of a (1e-4 .. 1.0) grid in 11 of
#: 24 fits -- a selection at a grid edge means the grid chose, not the data, and
#: this project has been caught by exactly that twice (see REGISTER cycle 4).
#: The extra values are cheap here: logreg on ~50k x few-hundred features.
#:
#: Extended again to 1e4 after the 60-subject re-run pinned `all_plus_stage` at
#: the 1e2 ceiling -- a genuine edge, not the flat tie-break the stage-only
#: baseline shows. Large C means "barely regularise", which is plausible for the
#: richest feature set, but the grid must be wide enough that the data can say
#: so rather than be truncated into saying so.
CVALS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
#: Learning rates swept for the boosted-tree variant. Sleep staging classically
#: rests on nonlinear band-ratio boundaries (delta/beta and friends), which a
#: linear model in envelope features cannot express, so a tree model is the
#: honest comparison rather than an upgrade smuggled in later.
LRATES = (0.05, 0.1, 0.2)
#: 128 envelope samples x 5 bands x 2 channels is 1280 raw features per window,
#: which is slow and mostly redundant because envelopes are smooth. Summarise each
#: (band, channel) trace over a few sub-windows instead.
N_CHUNKS = 8
#: Context length in envelope samples. Defaults to F11's window; --context-s
#: overrides it, because 12.8 s of history to predict 60 s ahead is short next to
#: how slowly sleep transitions actually develop.
CTX_L = L


def set_context(seconds: float) -> None:
    global CTX_L  # noqa: PLW0603 - the parameter under study
    CTX_L = int(round(seconds * ENV_FS))


def featurise(stack: np.ndarray) -> np.ndarray:
    """(n, C, L) -> (n, C * N_CHUNKS * 2): per-chunk mean and slope.

    Envelopes are slow, so raw 10 Hz samples are highly redundant. Mean captures
    level and slope captures direction, which is what a transition looks like.
    """
    n, c, ln = stack.shape
    chunks = np.array_split(np.arange(ln), N_CHUNKS)
    feats = []
    for idx in chunks:
        seg = stack[:, :, idx]
        feats.append(seg.mean(axis=2))
        t = np.arange(len(idx), dtype=float)
        t = (t - t.mean()) / (t.std() + 1e-9)
        feats.append((seg * t).mean(axis=2))          # slope, up to a constant
    return np.concatenate(feats, axis=1).reshape(n, -1).astype(np.float32)


def onehot(stages: np.ndarray) -> np.ndarray:
    return np.stack([(stages == c).astype(float) for c in CLASSES], axis=1)


def build(env_stack: np.ndarray, stage: np.ndarray, horizon: int):
    """Windows with strictly-future stage targets, plus the current stage."""
    last_in, tgt = CTX_L - 1, CTX_L - 1 + horizon
    assert tgt > last_in, f"horizon {horizon} would reintroduce overlap"
    n_t = env_stack.shape[1]
    starts = np.arange(0, n_t - (CTX_L + horizon), STRIDE)
    if len(starts) == 0:
        return None
    x = np.stack([env_stack[:, s:s + CTX_L] for s in starts])   # (n, C, CTX_L)
    y_then = np.array([stage[s + tgt] for s in starts])
    y_now = np.array([stage[s + last_in] for s in starts])
    keep = (y_then != "?") & (y_now != "?")
    return featurise(x[keep]), y_now[keep], y_then[keep]


def make_model(kind: str, hp: float, seed: int = SEED):
    if kind == "logreg":
        return LogisticRegression(C=hp, max_iter=400, class_weight="balanced")
    return HistGradientBoostingClassifier(
        learning_rate=hp, max_iter=300, early_stopping=True, validation_fraction=0.1,
        class_weight="balanced", random_state=seed)


def hparams(kind: str) -> tuple:
    return CVALS if kind == "logreg" else LRATES


def at_grid_edge(kind: str, best, label: str, scores=None) -> str:
    """Classify a hyper-parameter selection that landed on a grid edge.

    A selection at an edge usually means the grid, not the data, chose the
    hyper-parameter: the optimum may well lie outside what was searched. This
    project has twice mistaken an under-tuned baseline for a finding, so the
    condition is printed the moment it happens rather than left in the JSON for
    someone to notice later.

    But there are two ways to land on an edge and only one of them is a bug.
    The selection loop keeps the first strict improvement, so when *every* grid
    value scores the same the argmax falls on index 0 by default --- an edge by
    tie-break, not by optimum. That is what the stage-only model does: its
    one-hot features reproduce persistence exactly, so regularisation strength
    genuinely cannot matter. Extending the grid would not move it.

    Reporting both as the same warning would be crying wolf, and a warning that
    fires on healthy runs stops being read. So this returns one of:

      ``"no"``     not on an edge
      ``"tied"``   on an edge, but the validation curve is flat: harmless
      ``"edge"``   on an edge and strictly better than its neighbour: a bug

    Only ``"edge"`` blocks quoting the number.
    """
    grid = hparams(kind)
    if best not in (grid[0], grid[-1]):
        return "no"
    side = "bottom" if best == grid[0] else "top"

    # A flat curve means the tie-break chose it, not the data.
    if scores is not None and len(scores) > 1:
        span = float(np.max(scores) - np.min(scores))
        if span <= 1e-12:
            print(f"  note: {label} selected hp={best:g} at the {side} of the "
                  f"grid, but all {len(scores)} values score identically "
                  f"({float(np.max(scores)):.4f}) -- the tie-break chose it, "
                  f"not the grid edge; nothing to extend", flush=True)
            return "tied"

    print(f"  WARNING: {label} selected hp={best:g} at the {side} of the grid "
          f"[{grid[0]:g} .. {grid[-1]:g}] -- extend the grid before "
          f"quoting this number", flush=True)
    return "edge"


def evaluate(y_true, y_pred) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(CLASSES), zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "n": int(len(y_true)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/stage_forecast"))
    ap.add_argument("--sleep-edf", type=Path,
                    default=Path("/Users/aayu/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette"))
    ap.add_argument("--n-subjects", type=int, default=60)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--model", choices=("logreg", "hgb"), default="logreg")
    ap.add_argument("--context-s", type=float, default=L / ENV_FS,
                    help="Context window in seconds. The default 12.8 s is short "
                         "relative to a 60 s horizon -- sleep transitions build "
                         "over minutes, so this is the knob most likely to matter.")
    args = ap.parse_args()

    set_context(args.context_s)
    print(f"context {args.context_s:.0f} s = {CTX_L} envelope samples", flush=True)

    psg = sorted(args.sleep_edf.glob("*-PSG.edf"))[: args.n_subjects]
    n_samp = int(args.hours * 3600 * FS)
    band_names = list(BANDS)

    data: dict[int, dict] = {}
    for i, f in enumerate(psg):
        hyp = sorted(args.sleep_edf.glob(f"{f.name[:6]}*-Hypnogram.edf"))
        if not hyp:
            continue
        span = sleep_span(hyp[0])
        if span is None:
            continue
        ch = load_channels(f, ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"),
                           n_samp, span)
        if ch is None:
            continue
        eeg_tr, extra_tr = [], []
        for name in ("EEG Fpz-Cz", "EEG Pz-Oz"):
            for b in band_names:
                lo, hi = BANDS[b]
                eeg_tr.append(zscore(causal_envelope(ch[name], lo, hi)))
        for b in band_names:
            lo, hi = BANDS[b]
            extra_tr.append(zscore(causal_envelope(ch["EOG horizontal"], lo, hi)))
        # EMG is 1 Hz here and already a tone measure -- decimate, never band-pass.
        extra_tr.append(zscore(ch["EMG submental"][::DECIM]))

        k = min(min(len(t) for t in eeg_tr), min(len(t) for t in extra_tr))
        eeg_stack = np.stack([t[:k] for t in eeg_tr])
        all_stack = np.concatenate([eeg_stack, np.stack([t[:k] for t in extra_tr])], axis=0)
        st = stage_series(hyp[0], k, offset_s=span[0] / FS)
        if st is None or (st != "?").sum() < 0.5 * k:
            continue
        data[i] = {"eeg_only": eeg_stack, "eeg_eog_emg": all_stack, "stage": st}
        del ch, eeg_tr, extra_tr
        frac = {c: float(np.mean(st == c)) for c in CLASSES}
        print(f"  [{len(data)}] {f.name} {k/ENV_FS/3600:.1f} h  "
              + " ".join(f"{c}={frac[c]:.2f}" for c in CLASSES), flush=True)

    subs = sorted(data)
    n = len(subs)
    fit_s, sel_s, test_s = subs[: int(0.6 * n)], subs[int(0.6 * n): int(0.75 * n)], subs[int(0.75 * n):]
    print(f"split: {len(fit_s)} fit / {len(sel_s)} select / {len(test_s)} test", flush=True)

    out: dict = {}
    for inputs in ("stage_only", "eeg_only", "eeg_plus_stage", "all_plus_stage"):
        src = {"stage_only": "eeg_only", "eeg_only": "eeg_only",
               "eeg_plus_stage": "eeg_only", "all_plus_stage": "eeg_eog_emg"}[inputs]
        use_stage = inputs in ("stage_only", "eeg_plus_stage", "all_plus_stage")
        use_eeg = inputs != "stage_only"
        rows = []
        for h in HORIZONS:
            def stack(ss):
                parts = [build(data[s][src], data[s]["stage"], h) for s in ss]
                parts = [p for p in parts if p is not None]
                feats = np.concatenate([p[0] for p in parts])
                now = np.concatenate([p[1] for p in parts])
                then = np.concatenate([p[2] for p in parts])
                cols = ([feats] if use_eeg else []) + ([onehot(now)] if use_stage else [])
                return np.hstack(cols), now, then

            xf, _, yf = stack(fit_s)
            xs, _, ys = stack(sel_s)
            xt, now_t, yt = stack(test_s)

            best, best_score = None, -np.inf
            hp_scores = []
            for hp in hparams(args.model):
                m = make_model(args.model, hp)
                m.fit(xf, yf)
                sc = balanced_accuracy_score(ys, m.predict(xs))
                hp_scores.append(sc)
                if sc > best_score:
                    best, best_score = hp, sc
            clf = make_model(args.model, best)
            clf.fit(np.vstack([xf, xs]), np.concatenate([yf, ys]))
            pred = clf.predict(xt)                     # ONE model, then masked

            majority = max(CLASSES, key=lambda c: int(np.sum(yf == c)))
            trans = now_t != yt
            row = {
                "horizon_env_samples": h, "horizon_s": h / ENV_FS,
                "model": args.model, "selected_hp": best,
                "hp_at_grid_edge": at_grid_edge(
                    args.model, best, f"{inputs} staging h={h / ENV_FS:.0f}s",
                    scores=hp_scores),
                "transition_rate": float(trans.mean()),
                "all": {
                    "model": evaluate(yt, pred),
                    "persistence": evaluate(yt, now_t),
                    "majority": evaluate(yt, np.full_like(yt, majority)),
                },
            }
            if trans.sum() >= 50:
                row["transition"] = {
                    "model": evaluate(yt[trans], pred[trans]),
                    "persistence": evaluate(yt[trans], now_t[trans]),
                    "majority": evaluate(yt[trans], np.full_like(yt[trans], majority)),
                }
            rows.append(row)
            a, t = row["all"], row.get("transition")
            print(f"  {inputs:>14} {h/ENV_FS:>3.0f}s  ALL bal-acc model {a['model']['balanced_accuracy']:.3f} "
                  f"vs pers {a['persistence']['balanced_accuracy']:.3f} "
                  f"vs maj {a['majority']['balanced_accuracy']:.3f}"
                  + (f"   TRANSITION model {t['model']['balanced_accuracy']:.3f} "
                     f"vs pers {t['persistence']['balanced_accuracy']:.3f} "
                     f"(n={t['model']['n']}, {row['transition_rate']:.1%})" if t else ""),
                  flush=True)
        out[inputs] = rows

    # ---- Part two: binary transition detection -------------------------------
    print("\ntransition detection (will the stage change within h seconds?)", flush=True)
    trans_out: dict = {}
    for inputs in ("stage_only", "eeg_only", "eeg_plus_stage", "all_plus_stage"):
        src = {"stage_only": "eeg_only", "eeg_only": "eeg_only",
               "eeg_plus_stage": "eeg_only", "all_plus_stage": "eeg_eog_emg"}[inputs]
        use_stage = inputs in ("stage_only", "eeg_plus_stage", "all_plus_stage")
        use_eeg = inputs != "stage_only"
        rows = []
        for h in HORIZONS:
            def stack(ss):
                parts = [build(data[s][src], data[s]["stage"], h) for s in ss]
                parts = [p for p in parts if p is not None]
                feats = np.concatenate([p[0] for p in parts])
                now = np.concatenate([p[1] for p in parts])
                then = np.concatenate([p[2] for p in parts])
                cols = ([feats] if use_eeg else []) + ([onehot(now)] if use_stage else [])
                subj = np.concatenate([np.full(len(p[1]), s) for s, p in zip(ss, parts)])
                return np.hstack(cols), (now != then).astype(int), now, then, subj

            xf, yf, _, _, _ = stack(fit_s)
            xs, ys, _, _, _ = stack(sel_s)
            xt, yt, now_t, then_t, subj_t = stack(test_s)
            if yt.sum() < 50 or ys.sum() < 20:
                continue
            best, best_auc = None, -np.inf
            # Collected so at_grid_edge can tell a genuine boundary optimum from
            # a flat grid resolved by tie-break. Without the scores it has to
            # assume the worst and call every index-0 selection an edge, which
            # is what made the stage_only transition rows fire spuriously.
            hp_scores = []
            for hp in hparams(args.model):
                m = make_model(args.model, hp)
                m.fit(xf, yf)
                auc = roc_auc_score(ys, m.predict_proba(xs)[:, 1])
                hp_scores.append(auc)
                if auc > best_auc:
                    best, best_auc = hp, auc
            clf = make_model(args.model, best)
            clf.fit(np.vstack([xf, xs]), np.concatenate([yf, ys]))
            prob = clf.predict_proba(xt)[:, 1]
            base = float(yt.mean())

            # Label-noise stratification. N1 has inter-scorer kappa 0.24, so
            # transitions touching it are the least trustworthy labels in the set.
            # Negatives are kept as-is; only the POSITIVE class is filtered, so the
            # comparison is "same detector, more reliable positives".
            touches_n1 = (now_t == "N1") | (then_t == "N1")
            reliable = ~(yt.astype(bool) & touches_n1)
            n1_share = float((yt.astype(bool) & touches_n1).sum() / max(yt.sum(), 1))
            rel = {}
            if reliable.sum() > 100 and yt[reliable].sum() >= 50:
                rel = {
                    "roc_auc": float(roc_auc_score(yt[reliable], prob[reliable])),
                    "average_precision": float(average_precision_score(yt[reliable], prob[reliable])),
                    "base_rate": float(yt[reliable].mean()),
                    "n_positive": int(yt[reliable].sum()),
                }
            # Per-subject AUC. A pooled AUC can be carried by one subject, and a
            # per-subject reversal hiding inside a pooled win is exactly what the
            # BNCI 4 ms result turned out to be. Store both maskings.
            per_subject = []
            for s_id in sorted(set(subj_t.tolist())):
                m = subj_t == s_id
                if m.sum() < 100 or yt[m].sum() < 20 or yt[m].sum() == m.sum():
                    continue
                entry = {"subject": int(s_id), "n": int(m.sum()),
                         "roc_auc": float(roc_auc_score(yt[m], prob[m]))}
                mr = m & reliable
                if mr.sum() > 100 and 0 < yt[mr].sum() < mr.sum():
                    entry["roc_auc_excl_n1"] = float(roc_auc_score(yt[mr], prob[mr]))
                per_subject.append(entry)

            rows.append({
                "n1_share_of_transitions": n1_share,
                "excluding_n1_transitions": rel,
                "per_subject": per_subject,
                "horizon_s": h / ENV_FS, "model": args.model, "selected_hp": best,
                "hp_at_grid_edge": at_grid_edge(
                    args.model, best, f"{inputs} transition h={h / ENV_FS:.0f}s",
                    scores=hp_scores),
                "hp_scores": hp_scores,
                "roc_auc": float(roc_auc_score(yt, prob)),
                "average_precision": float(average_precision_score(yt, prob)),
                "base_rate": base, "lift_over_base": float(average_precision_score(yt, prob) / base),
                "n": int(len(yt)), "n_positive": int(yt.sum()),
            })
            r = rows[-1]
            ex = r.get("excluding_n1_transitions") or {}
            print(f"  {inputs:>14} {r['horizon_s']:>3.0f}s  AUC {r['roc_auc']:.3f}  "
                  f"AP {r['average_precision']:.3f} (base {base:.3f}, "
                  f"{r['lift_over_base']:.2f}x)  n+={r['n_positive']}"
                  + (f"   | excl-N1 AUC {ex['roc_auc']:.3f} AP {ex['average_precision']:.3f}"
                     f" (N1 was {r['n1_share_of_transitions']:.0%} of positives)" if ex else ""),
                  flush=True)
        trans_out[inputs] = rows

    # Paired per-subject comparison of each input set against stage_only. This is
    # the statistic a reviewer will ask for: not "is the pooled AUC higher" but
    # "is it higher in the same subjects, and could that be chance".
    paired: dict = {}
    ref = {r["horizon_s"]: {d["subject"]: d for d in r["per_subject"]}
           for r in trans_out.get("stage_only", [])}
    for inputs, rows in trans_out.items():
        if inputs == "stage_only":
            continue
        out_rows = []
        for r in rows:
            base = ref.get(r["horizon_s"], {})
            for key, lab in (("roc_auc", "all"), ("roc_auc_excl_n1", "excl_n1")):
                d = np.array([d[key] - base[d["subject"]][key]
                              for d in r["per_subject"]
                              if d["subject"] in base and key in d and key in base[d["subject"]]])
                if len(d) < 6:
                    continue
                rng = np.random.default_rng(SEED)
                boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(10_000)])
                lo95, hi95 = np.percentile(boot, [2.5, 97.5])
                out_rows.append({
                    "horizon_s": r["horizon_s"], "subset": lab, "n_subjects": len(d),
                    "mean_auc_gain": float(d.mean()),
                    "n_subjects_improved": int((d > 0).sum()),
                    "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
                    "ci95": [float(lo95), float(hi95)],
                    "ci_excludes_zero": bool(lo95 > 0 or hi95 < 0),
                })
                q = out_rows[-1]
                print(f"  paired {inputs:>14} vs stage_only {q['horizon_s']:>3.0f}s {lab:>7}: "
                      f"dAUC {q['mean_auc_gain']:+.4f}  {q['n_subjects_improved']}/{q['n_subjects']} up  "
                      f"CI[{q['ci95'][0]:+.4f},{q['ci95'][1]:+.4f}] p={q['wilcoxon_p']:.2e} "
                      f"{'EXCLUDES 0' if q['ci_excludes_zero'] else 'spans 0'}", flush=True)
        paired[inputs] = out_rows

    payload = {
        "schema": "kahlus.stage_forecast_sleepedf.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "can scored sleep stage be predicted 10-60 s before it occurs",
        "dataset": "Sleep-EDF sleep-cassette, sliced to the scored sleep period",
        "classes": list(CLASSES), "horizons_s": [h / ENV_FS for h in HORIZONS],
        "envelope_rate_hz": ENV_FS, "context_s": args.context_s,
        "context_env_samples": CTX_L, "stride": STRIDE,
        "features": f"per-(band,channel) chunk mean and slope, {N_CHUNKS} chunks",
        "model": args.model,
        "model_selection": "hyper-parameter chosen on held-out selection subjects, "
                           "never on the test subjects",
        "baselines": "persistence (stage now) and majority class from the fit split; "
                     "stage_only is persistence expressed as a fitted model and is the "
                     "reference every other input set must beat",
        "note": "persistence is the baseline that matters -- stages last minutes, so "
                "raw accuracy against a majority class would measure stage duration. "
                "The transition subset is where persistence is wrong by construction.",
        "n_subjects": n, "fit": fit_s, "select": sel_s, "test": test_s, "seed": SEED,
        "results": out,
        "transition_detection": {
            "task": "binary: does the scored stage change within h seconds",
            "metric": "ROC AUC and average precision; accuracy would be dominated "
                      "by the base rate since transitions are rare",
            "reference": "stage_only -- transition probability depends on the current "
                         "stage, so beating chance is too easy a bar",
            "label_noise_ceiling": {
                "overall_kappa": 0.76,
                "per_stage_kappa": {"W": 0.70, "N1": 0.24, "N2": 0.57, "N3": 0.57, "REM": 0.69},
                "raw_epoch_agreement": 0.82,
                "sources": [
                    "Lee YJ, Lee JY, Cho JH, Choi JH. Interrater reliability of sleep "
                    "stage scoring: a meta-analysis. J Clin Sleep Med. 2022;18(1):193-202.",
                    "Danker-Hopfe H, et al. Interrater reliability for sleep scoring "
                    "according to the Rechtschaffen & Kales and the new AASM standard. "
                    "J Sleep Res. 2009;18(1):74-84.",
                ],
                "implication": "a transition is defined by two labels, so at ~0.82 "
                               "per-epoch agreement roughly a third of scored transitions "
                               "may be scorer disagreement rather than physiology; AUC "
                               "against these labels is bounded well below 1",
            },
            "by_input": trans_out,
            "paired_vs_stage_only": paired,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
