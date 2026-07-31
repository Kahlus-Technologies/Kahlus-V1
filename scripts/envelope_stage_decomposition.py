"""F11b: is the 60-second envelope forecast just "the sleep stage did not change"?

F11 shows the low-gamma band-power envelope of Sleep-EDF Fpz-Cz is forecastable
60 s ahead, subject-held-out, and that removing submental EMG and EOG barely
touches it. The obvious next objection is that sleep stages last tens of minutes,
so a model asked to predict 60 s ahead can score well by learning nothing more
than "whatever is happening now is still happening then".

That objection deserves a number, not a paragraph. This splits the identical
predictions three ways using the scored hypnograms:

  ALL          every test window                        (what F11 reports)
  WITHIN       stage at t equals stage at t+h           (no transition occurred)
  TRANSITION   stage at t differs from stage at t+h     (a transition occurred)

Both readings are informative and neither is a failure:

  - skill only in WITHIN     -> the model tracks the current state and cannot
                                anticipate change. Still useful, but it is state
                                estimation and must not be called prediction of
                                transitions.
  - skill also in TRANSITION -> the envelope carries information about change
                                before it happens, which is the clinically
                                interesting claim and the one H2 was written for.

The model is NOT refit per subset. One model, one set of predictions, three
masks -- the same discipline as the overlap law, and for the same reason: if you
refit per subset, differences in capacity or sample size can explain a gap and
the comparison stops being about the signal.

  PYTHONPATH=src python3 scripts/envelope_stage_decomposition.py --out runs/envelope_stage
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envelope_horizon_sleepedf import (  # noqa: E402
    ALPHAS,
    BANDS,
    DECIM,
    ENV_FS,
    FS,
    L,
    SEED,
    causal_envelope,
    load_channels,
    sleep_span,
    ridge_fit,
    ridge_pred,
    score,
    zscore,
)

HORIZONS = (100, 300, 600)          # 10, 30, 60 s
STRIDE = 8
#: Collapse the scored stages to the states a clinician would act on. Stages 3
#: and 4 are both deep sleep and were merged in later scoring standards anyway.
STAGE_MAP = {
    "Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}


def stage_series(hyp_path: Path, n_env: int, offset_s: float = 0.0) -> np.ndarray | None:
    """Per-envelope-sample stage label, or None if unusable.

    Annotations are in seconds from FILE start, the envelope grid is ENV_FS Hz
    and begins at `offset_s` into the file, so every onset is shifted by that
    offset before being written. Getting this wrong silently misaligns every
    label -- and the offset is large here, because the recording is sliced to the
    scored sleep period, which begins 5-8 h in.
    """
    import mne

    try:
        ann = mne.read_annotations(hyp_path)
    except Exception as exc:  # noqa: BLE001 - a bad hypnogram is skipped, not fatal
        print(f"  skip {hyp_path.name}: {type(exc).__name__}: {exc}", flush=True)
        return None
    out = np.full(n_env, "?", dtype="<U4")
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        lab = STAGE_MAP.get(str(desc))
        if lab is None:
            continue
        a = int((onset - offset_s) * ENV_FS)
        b = int((onset + dur - offset_s) * ENV_FS)
        if b <= 0:
            continue
        a = max(a, 0)
        if a >= n_env:
            continue
        out[a:min(b, n_env)] = lab
    return out


def windows_with_stage(env: np.ndarray, stage: np.ndarray, horizon: int):
    last_in, tgt = L - 1, L - 1 + horizon
    assert tgt > last_in, f"horizon {horizon} would reintroduce overlap"
    starts = np.arange(0, len(env) - (L + horizon), STRIDE)
    if len(starts) == 0:
        empty = np.empty((0, L))
        return empty, np.empty(0), np.empty(0, dtype="<U4"), np.empty(0, dtype="<U4")
    x = np.stack([env[s:s + L] for s in starts])
    y = np.array([env[s + tgt] for s in starts])
    s_now = np.array([stage[s + last_in] for s in starts])
    s_then = np.array([stage[s + tgt] for s in starts])
    return x, y, s_now, s_then


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/envelope_stage"))
    ap.add_argument("--sleep-edf", type=Path,
                    default=Path("/Users/aayu/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette"))
    ap.add_argument("--n-subjects", type=int, default=45)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--band", type=str, default="low_gamma")
    args = ap.parse_args()

    lo, hi = BANDS[args.band]
    psg = sorted(args.sleep_edf.glob("*-PSG.edf"))[: args.n_subjects]
    n_samp = int(args.hours * 3600 * FS)

    env_by, stage_by = {}, {}
    for i, f in enumerate(psg):
        hyp = sorted(args.sleep_edf.glob(f"{f.name[:6]}*-Hypnogram.edf"))
        if not hyp:
            continue
        span = sleep_span(hyp[0])
        if span is None:
            continue
        ch = load_channels(f, ("EEG Fpz-Cz",), n_samp, span)
        if ch is None:
            continue
        sig = ch["EEG Fpz-Cz"]
        if len(sig) < 200 * FS:
            continue
        e = zscore(causal_envelope(sig, lo, hi))
        st = stage_series(hyp[0], len(e), offset_s=span[0] / FS)
        if st is None or (st != "?").sum() < 0.5 * len(st):
            continue
        env_by[i], stage_by[i] = e, st
        print(f"  [{len(env_by)}] {f.name}  {len(e)/ENV_FS/3600:.1f} h envelope", flush=True)

    subs = sorted(env_by)
    n = len(subs)
    fit_s, sel_s, test_s = subs[: int(0.6 * n)], subs[int(0.6 * n): int(0.75 * n)], subs[int(0.75 * n):]
    print(f"split: {len(fit_s)} fit / {len(sel_s)} select / {len(test_s)} test", flush=True)

    rows = []
    for h in HORIZONS:
        def stack(ss):
            parts = [windows_with_stage(env_by[s], stage_by[s], h) for s in ss]
            return (np.concatenate([p[0] for p in parts]),
                    np.concatenate([p[1] for p in parts]),
                    np.concatenate([p[2] for p in parts]),
                    np.concatenate([p[3] for p in parts]))

        xf, yf, _, _ = stack(fit_s)
        xs, ys, _, _ = stack(sel_s)
        xt, yt, now, then = stack(test_s)
        best = min(ALPHAS, key=lambda a: float(np.mean((ys - ridge_pred(xs, ridge_fit(xf, yf, a))) ** 2)))
        w = ridge_fit(np.vstack([xf, xs]), np.concatenate([yf, ys]), best)
        pred = ridge_pred(xt, w)                       # ONE model, ONE prediction
        pers = xt[:, -1]

        valid = (now != "?") & (then != "?")
        masks = {
            "all": valid,
            "within_stage": valid & (now == then),
            "transition": valid & (now != then),
        }
        out = {"horizon_env_samples": h, "horizon_s": h / ENV_FS, "selected_alpha": best}
        for name, m in masks.items():
            if m.sum() < 50:
                out[name] = {"n": int(m.sum()), "note": "too few windows to score"}
                continue
            out[name] = {
                "n": int(m.sum()), "frac_of_valid": float(m.sum() / valid.sum()),
                **score(yt[m], pred[m]),
                "persistence_mse": float(np.mean((yt[m] - pers[m]) ** 2)),
                "trace_mean_mse": float(np.mean((yt[m] - yt[m].mean()) ** 2)),
            }
        rows.append(out)
        a, wi, tr = out["all"], out["within_stage"], out["transition"]
        print(f"  {h/ENV_FS:>3.0f}s  ALL r={a['pearsonr']:+.3f} (n={a['n']})   "
              f"WITHIN r={wi.get('pearsonr', float('nan')):+.3f} (n={wi['n']}, "
              f"{wi.get('frac_of_valid', 0):.0%})   "
              f"TRANSITION r={tr.get('pearsonr', float('nan')):+.3f} (n={tr['n']}, "
              f"{tr.get('frac_of_valid', 0):.0%})", flush=True)

    payload = {
        "schema": "kahlus.envelope_stage_decomposition.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "is the 60 s envelope forecast merely 'the stage did not change'",
        "band": args.band, "band_hz": [lo, hi],
        "envelope_rate_hz": ENV_FS, "window_env_samples": L, "decim": DECIM,
        "protocol": "one model per horizon, one set of predictions, three scoring masks; "
                    "the model is never refit per subset",
        "n_subjects": n, "fit": fit_s, "select": sel_s, "test": test_s,
        "span": "sliced to the scored sleep period (+/- 10 min); slicing from file "
                "start yields ~100% wake in this corpus",
        "seed": SEED, "horizons": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
