"""Low-cost montage stress test for arousal detection (issue #97).

A single-channel C4 arousal *detector* (not forecaster) is trained on clean PSG
from one cohort and tested on the other cohort's EEG after headband-like
degradation (montage_degrade.LEVELS). Event-level scoring: detected onsets are
matched one-to-one to scored arousal onsets within TOL_S.

Per second t during scored sleep: features from the 3 s of EEG ending at t
(eeg_features) plus the same features causally z-scored within the night. Label:
t lies inside any scored arousal. Threshold picked on the training cohort to
maximize event F1, then frozen.

Usage: python scripts/run_montage_stress.py --mesa ~/data/nsrr/mesa --shhs ~/data/nsrr/shhs --out artifacts/montage_stress_dev25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from neurotwin.adapters.nsrr import (
    BAD_NIGHT_ERRORS,
    SLEEP_STAGES,
    SkippedNights,
    discover_nsrr_recordings,
    event_kind,
    load_nsrr_epoch_matrix,
    mesa_fit_allowed,
    nsrr_root,
    parse_nsrr_events,
    sleep_stage_epochs,
)
from neurotwin.forecastability.heart_arousal import COHORTS, causal_z, eeg_features, make_model
from neurotwin.forecastability.montage_degrade import LEVELS, degrade, detected_onsets, match_events

TOL_S = 3.0
WINDOW_S = 3.0


def night_rows(eeg_uv: np.ndarray, fs: float, events, level: str, seed: int) -> dict:
    x = degrade(eeg_uv, fs, level, seed=seed)
    n_sec = int(x.size / fs)
    asleep = np.zeros(n_sec, dtype=bool)
    for s, e, c in sleep_stage_epochs(events):
        if c in SLEEP_STAGES:
            asleep[int(s) : min(int(e), n_sec)] = True
    inside = np.zeros(n_sec, dtype=bool)
    onsets = []
    for c, s, d in events:
        if event_kind(c) == "arousal":
            inside[int(s) : min(int(np.ceil(s + d)), n_sec)] = True  # second t is labeled if it starts inside [s, s+d)
            if int(s) < n_sec and asleep[int(s)]:
                onsets.append(s)
    t = np.arange(int(WINDOW_S), n_sec, dtype=np.float64)
    f = eeg_features(x, fs, t, window_s=WINDOW_S)
    feats = np.concatenate([f, causal_z(f)], axis=1)
    keep = asleep[t.astype(int)]
    return {"t": t[keep], "x": feats[keep], "y": inside[t.astype(int)][keep].astype(int), "onsets": np.asarray(onsets)}


def read_night(cohort: str, rec) -> tuple[np.ndarray, float, list]:
    """One night's C4 EEG in microvolts, its sampling rate and its scored events."""
    m = load_nsrr_epoch_matrix(rec.edf_path, channel_groups={"eeg": COHORTS[cohort]["eeg_labels"]})
    return m["epochs"]["eeg"][:, :, 0].ravel() * 1e6, m["sfreq"], parse_nsrr_events(rec.xml_path)


def load_clean(cohort: str, root: Path) -> tuple[list, list[dict], SkippedNights]:
    """Clean-montage rows of every usable night, reading one night at a time.

    Only the rows stay in memory, never a cohort's raw EEG. Night i keeps degrade
    seed i, its position among the usable nights. Returns the usable recordings,
    their rows and the skipped nights.
    """
    recs, clean, skipped = [], [], SkippedNights(cohort)
    for rec in discover_nsrr_recordings(root, dataset=cohort):
        try:
            clean.append(night_rows(*read_night(cohort, rec), "clean", len(recs)))
        except BAD_NIGHT_ERRORS as exc:
            skipped.add(rec, exc)
            continue
        recs.append(rec)
    if not recs:
        raise RuntimeError(f"no readable {cohort} nights ({len(skipped)} skipped)")
    return recs, clean, skipped


def score_levels(cohort: str, recs: list, clean: list[dict], clf) -> dict[str, tuple[list[dict], list[np.ndarray]]]:
    """Each level's night timing and detector scores on the test cohort.

    Re-reads each test night once and degrades it at every level, so one night of raw
    EEG is in memory at a time. The results are the same as degrading all nights first.
    """
    out = {level: ([], []) for level in LEVELS}
    for i, (rec, night) in enumerate(zip(recs, clean)):
        eeg, fs, events = read_night(cohort, rec)
        for level in LEVELS:
            n = night if level == "clean" else night_rows(eeg, fs, events, level, i)
            out[level][0].append({"t": n["t"], "onsets": n["onsets"]})
            out[level][1].append(clf.predict_proba(n["x"])[:, 1])
    return out


def score_nights(nights: list[dict], prob: list[np.ndarray], thr: float) -> dict:
    # Offset each night far apart on one time axis so matches never cross nights.
    off, d_all, t_all = 0.0, [], []
    for n, p in zip(nights, prob):
        d_all.append(detected_onsets(p, n["t"], thr) + off)
        t_all.append(n["onsets"] + off)
        off += (n["t"][-1] if n["t"].size else 0.0) + 1e5
    m = match_events(np.concatenate(d_all), np.concatenate(t_all), TOL_S)
    hours = sum(n["t"].size for n in nights) / 3600.0
    m["false_detections_per_hour"] = (m["n_detected"] - m["tp"]) / hours
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesa", type=Path, default=nsrr_root("mesa"))
    ap.add_argument("--shhs", type=Path, default=nsrr_root("shhs"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    loaded = {c: load_clean(c, getattr(a, c)) for c in ("mesa", "shhs")}
    recs = {c: r for c, (r, _, _) in loaded.items()}
    clean = {c: rows for c, (_, rows, _) in loaded.items()}
    report = {"tol_s": TOL_S, "levels": list(LEVELS), "results": {},
              "nights": {c: len(r) for c, r in recs.items()},
              "skipped": {c: s.to_json() for c, (_, _, s) in loaded.items()}}
    for src, dst in (("mesa", "shhs"), ("shhs", "mesa")):
        if src == "mesa" and not mesa_fit_allowed():
            continue  # final mode: held-out MESA is test-only
        tr = clean[src]
        x, y = np.concatenate([n["x"] for n in tr]), np.concatenate([n["y"] for n in tr])
        ok = ~np.isnan(x).all(axis=1)
        clf = make_model(0).fit(x[ok], y[ok])
        p_tr = [clf.predict_proba(n["x"])[:, 1] for n in tr]
        thr = max(np.linspace(0.05, 0.9, 18), key=lambda th: score_nights(tr, p_tr, th)["f1"])
        rows = {"threshold": float(thr), "train_f1": score_nights(tr, p_tr, thr)["f1"]}
        scored = score_levels(dst, recs[dst], clean[dst], clf)
        for level in LEVELS:
            rows[level] = score_nights(*scored[level], thr)
            r = rows[level]
            print(f"{src}->{dst} {level:17s} F1 {r['f1']:.3f} P {r['precision']:.3f} R {r['recall']:.3f} "
                  f"timing {r['median_abs_timing_error_s']:.2f}s FD/h {r['false_detections_per_hour']:.1f}")
        report["results"][f"{src}_to_{dst}"] = rows
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
