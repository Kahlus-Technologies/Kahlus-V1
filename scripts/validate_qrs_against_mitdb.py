"""Validate the fast QRS detector against EXPERT-ANNOTATED beats (MIT-BIH Arrhythmia).

The plan validated qrs_fast against wfdb's xqrs. xqrs is not ground truth -- it is
another heuristic, so agreement between the two is not evidence either is correct,
and if both are wrong the check passes.

mitdb carries cardiologist-annotated beat locations. Metrics follow ANSI/AAMI EC38:
a detection matches an annotated beat within 150 ms; Se = TP/(TP+FN),
PPV = TP/(TP+FP). We also report the derived-HR correlation, because for this
project the rate SERIES is what enters Z -- a detector can match beats well and
still shred the rate series via split RR intervals (this is exactly what the
98.9%-match / r=0.148 row in the plan was showing).
"""
import sys
from pathlib import Path
import numpy as np
import wfdb

sys.path.insert(0, "src")
from neurotwin.adapters.qrs_fast import detect_qrs

# AAMI beat-annotation symbols; everything else (rhythm, noise, artifact) is not a beat.
BEAT_SYMS = set("NLRejAaJSVEFP/fQ")
MATCH_MS = 150.0

dest = Path(sys.argv[1])
recs = sorted(p.stem for p in dest.glob("*.atr"))
print(f"{'rec':>5} {'true':>6} {'det':>6} {'Se%':>7} {'PPV%':>7} {'HR r':>7} {'med|dHR|':>9}")
rows = []
for r in recs:
    rec = wfdb.rdrecord(str(dest / r))
    ann = wfdb.rdann(str(dest / r), "atr")
    fs = float(rec.fs)
    truth = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMS], dtype=float)
    ecg = np.nan_to_num(rec.p_signal[:, 0].astype(np.float64))
    det = np.asarray(detect_qrs(ecg, fs), dtype=float)
    if len(det) == 0 or len(truth) == 0:
        print(f"{r:>5} {len(truth):>6} {len(det):>6}   no detections")
        continue

    tol = MATCH_MS / 1000.0 * fs
    # greedy nearest-match, each truth beat consumed at most once
    used = np.zeros(len(det), dtype=bool)
    tp = 0
    for t in truth:
        d = np.abs(det - t)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    fn = len(truth) - tp
    fp = len(det) - tp
    se = 100.0 * tp / (tp + fn)
    ppv = 100.0 * tp / (tp + fp)

    # derived HR series on a 1 Hz grid, the thing that actually enters Z
    def hr_grid(beats):
        t = beats / fs
        if len(t) < 3:
            return None
        rate = 60.0 / np.diff(t)
        n = int(len(ecg) / fs)
        return np.interp(np.arange(n), t[1:], rate, left=rate[0], right=rate[-1])
    a, b = hr_grid(truth), hr_grid(det)
    rr = float(np.corrcoef(a, b)[0, 1]) if a is not None and b is not None else float("nan")
    dhr = float(np.median(np.abs(a - b))) if a is not None and b is not None else float("nan")
    rows.append((se, ppv, rr, dhr))
    print(f"{r:>5} {len(truth):>6} {len(det):>6} {se:>7.2f} {ppv:>7.2f} {rr:>7.3f} {dhr:>9.2f}")

if rows:
    a = np.array(rows)
    print(f"\n  n={len(rows)} records")
    print(f"  Se   mean {a[:,0].mean():6.2f}%  min {a[:,0].min():6.2f}%")
    print(f"  PPV  mean {a[:,1].mean():6.2f}%  min {a[:,1].min():6.2f}%")
    print(f"  HR r mean {a[:,2].mean():6.3f}   min {a[:,2].min():6.3f}   "
          f"records with r<0.95: {(a[:,2]<0.95).sum()}/{len(rows)}")
    print(f"  median |dHR| across records: {np.median(a[:,3]):.2f} bpm")
