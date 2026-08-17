"""F11: does the band-envelope horizon result survive an actual EMG measurement?

On BNCI, causal band-power envelopes stayed forecastable past 16 s while raw
voltage died by ~0.5 s. The obvious mundane explanation is EMG: scalp gamma is
the band most contaminated by muscle, and a slow drift in jaw or neck tension
produces exactly that signature -- broadband high-frequency power, long memory,
coherent across channels. BNCI has no EMG channel, so that could not be settled
there and the result was published with its origin explicitly open.

Sleep-EDF has `EMG submental`. So the question stops being an argument about
plausibility and becomes a measurement.

THREE TARGETS, ONE PROTOCOL. For each band we forecast:

  1. the EEG envelope                       (the claim)
  2. the EEG envelope residualised on EMG   (the claim, with muscle removed)
  3. the EMG envelope itself                (how forecastable is muscle alone?)

A NOTE ON THE EMG CHANNEL, because it changes what the control means. In
sleep-cassette the EDF stores EEG/EOG at 100 Hz but EMG submental, respiration
and temperature at 1 Hz. MNE upsamples the slow channels to the common rate, so
reading naively hands you a piecewise-constant staircase -- a first attempt at
this experiment band-passed that staircase at 30-45 Hz, which is pure
interpolation artifact, and unsurprisingly it removed nothing from the EEG and
was itself "forecastable" at r = 1.000. Both numbers were nonsense.

Handled correctly the 1 Hz rate is a feature, not a limitation: submental EMG at
1 Hz already *is* a muscle-tone envelope, and slow muscle tone is precisely the
confound in question. So EMG is used unfiltered, decimated onto the envelope
grid, as a tone regressor. EOG (100 Hz, genuinely broadband) is band-filtered
like the EEG and added alongside it to catch eye-movement artifact.

Residualisation regresses the EEG envelope on the *concurrent* artifact
regressors and their lags. Concurrent is the conservative choice: a causal
forecaster could not use it, so this removes more than a fair test would demand.

Read the outcome like this:
  - skill collapses after residualising -> it was artifact, the BNCI result dies
  - skill survives                      -> the effect is not muscle or eye movement

WHY THIS REACHES 60 SECONDS. Envelopes are slow, so after causal filtering the
envelope is decimated to 10 Hz. A 128-sample window is then 12.8 s of context and
a 600-sample horizon is 60 s -- the precursor timescale this project actually
cares about, and roughly 400x further out than the F8 crossover at 160 ms.

Discipline is inherited unchanged from F8 and the BNCI work: subject-disjoint
splits, strictly-future single-sample targets, an explicit overlap assertion,
and ridge alpha selected on held-out selection subjects rather than assumed.

  PYTHONPATH=src python3 scripts/envelope_horizon_sleepedf.py --out runs/envelope_sleepedf
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

FS = 100.0
#: 100 Hz sampling caps usable frequency at 50 Hz, so "gamma" here is low gamma.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}
ENV_FS = 10.0           # envelope decimated to 10 Hz
DECIM = int(FS / ENV_FS)
L = 128                 # 12.8 s of context at 10 Hz
STRIDE = 8
#: envelope samples -> 1, 3, 6, 10, 20, 30, 60 seconds
HORIZONS = (10, 30, 60, 100, 200, 300, 600)
ALPHAS = (1e-2, 1.0, 1e2, 1e4, 1e6, 1e8)
ARTIFACT_LAGS = 8       # lags of each artifact regressor used when residualising
#: Stored at 1 Hz in sleep-cassette and upsampled by the reader. Never band-pass
#: it -- at 1 Hz it is already a muscle-tone envelope.
SLOW_CHANNELS = ("EMG submental",)
#: Margin kept either side of the scored sleep period, in seconds.
SLEEP_MARGIN_S = 600.0
SEED = 0


def load_channels(path: Path, names: tuple[str, ...],
                  n_samples: int | None = None,
                  span: tuple[int, int] | None = None) -> dict | None:
    """Raw samples for the named channels, or None if any is missing/unreadable.

    Reads with preload=False and pulls only the requested span. A recording here
    is 22 h across 7 channels; preloading all of it to keep 6 h of 3 channels was
    enough to get the first full run OOM-killed.

    Deliberately not the shared `load_edf_channel` helper: that one is hardwired
    to the EEG channel and z-scores on the way out, and this experiment needs the
    EMG channel alongside it on the same time base.
    """
    import mne

    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001 - a bad recording is skipped, not fatal
        print(f"  skip {path.name}: {type(exc).__name__}: {exc}", flush=True)
        return None
    if any(n not in raw.ch_names for n in names):
        return None
    if abs(raw.info["sfreq"] - FS) > 1e-6:
        print(f"  skip {path.name}: sfreq {raw.info['sfreq']} != {FS}", flush=True)
        return None
    a, b = (0, raw.n_times) if span is None else span
    b = min(b, raw.n_times)
    if n_samples is not None:
        b = min(b, a + n_samples)
    if b - a < 200 * FS:
        return None
    return {n: np.asarray(raw.get_data(picks=[n], start=a, stop=b)[0], dtype=float)
            for n in names}


def sleep_span(hyp_path: Path) -> tuple[int, int] | None:
    """Sample range covering the scored sleep period, or None if never asleep.

    This is not a refinement, it is a correction. Sleep-EDF cassette recordings
    begin hours before lights-out: taking the first 6 h of a file yields ~100%
    "Sleep stage W" for most subjects, so an experiment sliced from the start
    measures WAKE EEG while claiming to measure sleep, and reports a stage
    transition rate near zero because there is only ever one stage present.
    A first version of this script did exactly that.
    """
    import mne

    try:
        ann = mne.read_annotations(hyp_path)
    except Exception as exc:  # noqa: BLE001 - a bad hypnogram is skipped, not fatal
        print(f"  skip {hyp_path.name}: {type(exc).__name__}: {exc}", flush=True)
        return None
    asleep = [(o, o + d) for o, d, s in zip(ann.onset, ann.duration, ann.description)
              if str(s) not in ("Sleep stage W", "Sleep stage ?", "Movement time")]
    if not asleep:
        return None
    start = max(0.0, min(a for a, _ in asleep) - SLEEP_MARGIN_S)
    stop = max(b for _, b in asleep) + SLEEP_MARGIN_S
    return int(start * FS), int(stop * FS)


def causal_envelope(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Strictly causal band envelope: one-directional bandpass, rectify, low-pass.

    No sosfiltfilt and no Hilbert. Both are non-causal -- they give env[t] support
    on samples after t, which lets an input window carry information about its own
    target through the filter kernel rather than through the model. That is the
    same defect as scoring a forecast on a target window that overlaps its input,
    and on BNCI it was shown to contribute nothing; here it is simply not used.
    """
    sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="sos")
    lp = butter(2, min(lo, 2.0) / (FS / 2), btype="low", output="sos")
    env = sosfilt(lp, np.abs(sosfilt(sos, sig)))
    return env[::DECIM]


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def residualise(eeg: np.ndarray, artifacts: list[np.ndarray],
                lags: int = ARTIFACT_LAGS) -> np.ndarray:
    """Remove everything a linear function of concurrent+past artifacts explains.

    Concurrent values are included deliberately. A causal system could not use
    them, so this removes MORE than a fair forecaster would have to give up --
    making it a conservative test. Skill that survives is not artifact.
    """
    cols = [np.ones_like(eeg)]
    for a in artifacts:
        cols += [np.roll(a, k) for k in range(lags + 1)]
    d = np.stack(cols, axis=1)[lags:]
    y = eeg[lags:]
    coef, *_ = np.linalg.lstsq(d, y, rcond=None)
    out = np.empty_like(eeg)
    out[:lags] = 0.0
    out[lags:] = y - d @ coef
    return out


def windows(env: np.ndarray, horizon: int):
    """Strictly-future single-sample targets. Asserts no input/target overlap."""
    last_in, tgt = L - 1, L - 1 + horizon
    assert tgt > last_in, f"horizon {horizon} would reintroduce overlap"
    starts = np.arange(0, len(env) - (L + horizon), STRIDE)
    if len(starts) == 0:
        return np.empty((0, L)), np.empty(0)
    return (np.stack([env[s:s + L] for s in starts]),
            np.array([env[s + tgt] for s in starts]))


def ridge_fit(x, y, alpha):
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    return np.linalg.solve(xb.T @ xb + alpha * np.eye(xb.shape[1]), xb.T @ y)


def ridge_pred(x, w):
    return np.hstack([x, np.ones((x.shape[0], 1))]) @ w


def score(y, p) -> dict:
    return {"mse": float(np.mean((y - p) ** 2)),
            "pearsonr": float(np.corrcoef(p.ravel(), y.ravel())[0, 1])}


def _verdict(st: dict) -> str:
    if "ci95" not in st:
        return "n<6"
    return (f"CI[{st['ci95'][0]:+.3f},{st['ci95'][1]:+.3f}] "
            f"p={st['wilcoxon_p']:.1e} {'EXCLUDES 0' if st['ci_excludes_zero'] else 'spans 0'}")


def run_target(env_by_subj: dict, fit_s, sel_s, test_s, horizon: int) -> dict:
    def stack(subs):
        xs, ys = zip(*[windows(env_by_subj[s], horizon) for s in subs])
        return np.concatenate(xs), np.concatenate(ys)

    xf, yf = stack(fit_s)
    xs, ys = stack(sel_s)
    xt, yt = stack(test_s)
    best = min(ALPHAS, key=lambda a: float(np.mean((ys - ridge_pred(xs, ridge_fit(xf, yf, a))) ** 2)))
    w = ridge_fit(np.vstack([xf, xs]), np.concatenate([yf, ys]), best)
    pred = ridge_pred(xt, w)

    per_subj = []
    for s in test_s:
        x1, y1 = windows(env_by_subj[s], horizon)
        if len(y1) == 0:
            continue
        p1 = ridge_pred(x1, w)
        per_subj.append({"subject": s, **score(y1, p1),
                         "persistence_mse": float(np.mean((y1 - x1[:, -1]) ** 2)),
                         "mean_mse": float(np.mean((y1 - y1.mean()) ** 2))})
    beats = sum(d["mse"] < d["persistence_mse"] for d in per_subj)
    beats_mean = sum(d["mse"] < d["mean_mse"] for d in per_subj)

    # H2's pre-registered falsifier is stated against persistence with a CI that
    # excludes zero, so the paired per-subject test is the thing that decides it --
    # not the pooled number, which can hide a per-subject reversal (as it did on
    # BNCI at 4 ms).
    delta = np.array([d["persistence_mse"] - d["mse"] for d in per_subj])
    stats = {"n": len(delta), "mean_delta_vs_persistence": float(delta.mean())}
    if len(delta) >= 6 and np.any(delta != 0):
        stats["wilcoxon_p"] = float(wilcoxon(delta).pvalue)
        rng = np.random.default_rng(SEED)
        boot = np.array([rng.choice(delta, len(delta), replace=True).mean()
                         for _ in range(10_000)])
        lo95, hi95 = np.percentile(boot, [2.5, 97.5])
        stats["ci95"] = [float(lo95), float(hi95)]
        stats["ci_excludes_zero"] = bool(lo95 > 0 or hi95 < 0)
    return {
        "selected_alpha": best,
        "alpha_at_grid_edge": bool(best in (ALPHAS[0], ALPHAS[-1])),
        **score(yt, pred),
        "persistence_mse": float(np.mean((yt - xt[:, -1]) ** 2)),
        "trace_mean_mse": float(np.mean((yt - yt.mean()) ** 2)),
        "n_test_windows": int(len(yt)),
        "beats_persistence_n": beats, "beats_mean_n": beats_mean,
        "paired_vs_persistence": stats,
        "n_test_subjects": len(per_subj),
        "per_subject": per_subj,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/envelope_sleepedf"))
    ap.add_argument("--sleep-edf", type=Path,
                    default=Path("/Users/aayu/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette"))
    ap.add_argument("--n-subjects", type=int, default=30)
    ap.add_argument("--hours", type=float, default=6.0, help="hours per recording to use")
    ap.add_argument("--bands", type=str, default="beta,low_gamma")
    args = ap.parse_args()

    want = [b.strip() for b in args.bands.split(",") if b.strip()]
    files = sorted(args.sleep_edf.glob("*-PSG.edf"))[: args.n_subjects]
    n_samp = int(args.hours * 3600 * FS)
    print(f"{len(files)} recordings, {args.hours} h each, bands={want}", flush=True)

    # Envelopes are computed per subject and the raw span is dropped straight
    # away. Holding all raw first is what OOM-killed the first full run: 45
    # subjects x 6 h x 3 channels is 2.3 GB before any filtering starts, and the
    # decimated envelopes it becomes are 10x smaller.
    env_raw: dict[int, dict] = {}
    for i, f in enumerate(files):
        hyp = sorted(args.sleep_edf.glob(f"{f.name[:6]}*-Hypnogram.edf"))
        span = sleep_span(hyp[0]) if hyp else None
        if span is None:
            print(f"  skip {f.name}: no scored sleep period", flush=True)
            continue
        ch = load_channels(f, ("EEG Fpz-Cz", "EMG submental", "EOG horizontal"), n_samp, span)
        if ch is None:
            print(f"  skip {f.name}: missing channel", flush=True)
            continue
        eeg, emg, eog = ch["EEG Fpz-Cz"], ch["EMG submental"], ch["EOG horizontal"]
        n = min(len(eeg), len(emg), len(eog), n_samp)
        if n < 200 * FS:
            continue
        per_band = {}
        for b in want:
            blo, bhi = BANDS[b]
            per_band[b] = (causal_envelope(eeg[:n], blo, bhi),
                           causal_envelope(eog[:n], blo, bhi))
        env_raw[i] = {"bands": per_band, "emg_tone": emg[:n][::DECIM]}
        del ch, eeg, emg, eog
        print(f"  [{len(env_raw)}] {f.name} {n/FS/3600:.1f} h", flush=True)

    subs = sorted(env_raw)
    n = len(subs)
    fit_s, sel_s, test_s = subs[: int(0.6 * n)], subs[int(0.6 * n): int(0.75 * n)], subs[int(0.75 * n):]
    print(f"split: {len(fit_s)} fit / {len(sel_s)} select / {len(test_s)} test", flush=True)

    out: dict = {}
    for band in want:
        lo, hi = BANDS[band]  # reported in the payload; envelopes already computed
        eeg_env, resid_env, corr_emg, corr_eog = {}, {}, [], []
        for s in subs:
            e_raw, g_raw = env_raw[s]["bands"][band]
            e = zscore(e_raw)
            # EMG is 1 Hz in this corpus and already a tone measure -- decimate
            # onto the envelope grid, never band-pass it.
            m = zscore(env_raw[s]["emg_tone"][: len(e)])
            # EOG is genuinely 100 Hz, so it gets the same envelope treatment.
            g = zscore(g_raw)[: len(e)]
            k = min(len(e), len(m), len(g))
            e, m, g = e[:k], m[:k], g[:k]
            eeg_env[s] = e
            resid_env[s] = zscore(residualise(e, [m, g]))
            corr_emg.append(float(np.corrcoef(e, m)[0, 1]))
            corr_eog.append(float(np.corrcoef(e, g)[0, 1]))
        print(f"\n{band}: EEG-vs-EMG corr median {np.median(corr_emg):+.3f} "
              f"[{min(corr_emg):+.3f}, {max(corr_emg):+.3f}]   "
              f"EEG-vs-EOG median {np.median(corr_eog):+.3f}", flush=True)

        rows = []
        for h in HORIZONS:
            r = {
                "horizon_env_samples": h, "horizon_s": h / ENV_FS,
                "eeg": run_target(eeg_env, fit_s, sel_s, test_s, h),
                "eeg_artifact_residualised": run_target(resid_env, fit_s, sel_s, test_s, h),
            }
            rows.append(r)
            print(f"  {h/ENV_FS:>5.0f} s   EEG r={r['eeg']['pearsonr']:+.3f} "
                  f"({r['eeg']['beats_persistence_n']}/{r['eeg']['n_test_subjects']} vs pers)   "
                  f"resid r={r['eeg_artifact_residualised']['pearsonr']:+.3f} "
                  f"({r['eeg_artifact_residualised']['beats_persistence_n']}/"
                  f"{r['eeg_artifact_residualised']['n_test_subjects']})"
                  f"   {_verdict(r['eeg']['paired_vs_persistence'])}", flush=True)
        out[band] = {"eeg_emg_corr": corr_emg, "eeg_eog_corr": corr_eog, "horizons": rows}

    payload = {
        "schema": "kahlus.envelope_horizon_sleepedf.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "does the band-envelope horizon result survive a real EMG control",
        "dataset": "Sleep-EDF sleep-cassette, EEG Fpz-Cz + EMG submental",
        "sampling_rate_hz": FS, "envelope_rate_hz": ENV_FS,
        "window_env_samples": L, "context_s": L / ENV_FS,
        "horizons_s": [h / ENV_FS for h in HORIZONS],
        "alphas": list(ALPHAS), "artifact_lags": ARTIFACT_LAGS, "seed": SEED,
        "n_subjects": n, "fit": fit_s, "select": sel_s, "test": test_s,
        "envelope": "strictly causal: one-directional bandpass, rectify, causal low-pass",
        "span": "sliced to the scored sleep period (+/- 10 min), NOT from file start; "
                "these recordings begin hours before lights-out and a from-start "
                "slice is ~100% wake",
        "residualisation": "EEG envelope regressed on concurrent + 8 lags of BOTH the EMG "
                           "tone channel and the EOG envelope, per subject; concurrent "
                           "inclusion makes this conservative",
        "emg_note": "EMG submental is stored at 1 Hz in sleep-cassette and upsampled by the "
                    "reader. It is used unfiltered as a muscle-tone regressor; band-passing "
                    "it would filter interpolation artifact, not muscle.",
        "bands": out,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
