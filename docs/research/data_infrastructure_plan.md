# Kahlus Data Infrastructure — State & Plan

Last updated: 2026-07-26 · Supersedes ad hoc fetch notes · [AGENTS.md](../../AGENTS.md)

---

## TL;DR

**Three tiers, all free, each holding what it's actually good at:**

| Tier | Contents | Size | Home | Cost | Control |
| --- | --- | ---: | --- | ---: | --- |
| Raw public archives | `.mat` / EDF recordings | 133 GB+ | AWS Open Data S3 (already mirrored) | $0 | Immutable, can't be deleted or billed |
| Derived features | epoch arrays, manifests | ~2 GB | Hugging Face dataset repo (or R2) | $0 | **Yours** |
| Code + evidence | adapters, gates, artifacts | tiny | GitHub | $0 | **Yours** |

**Do not build your own S3 bucket for raw data.** The fast server you wanted already exists and is already AWS — see [The two PhysioNet servers](#the-two-physionet-servers). Re-hosting costs money, adds egress billing, and gains nothing.

**Chapman A100 flow:** clone repo → `aws s3 cp --no-sign-request` the raw data (~90 min) → extract → gate → push artifacts back. No Hetzner, no rsync, no laptop staging.

---

## 1. The two PhysioNet servers

This is the single most important operational fact discovered this session.

| Host | Throughput | Notes |
| --- | ---: | --- |
| `physionet.org` (direct HTTP) | **~100 KB/s** | Per-IP throttle. Parallel connections split the same cap — aria2c with 16 connections measured *worse* at 19 KB/s. |
| `physionet-open.s3.amazonaws.com` | **~25 MB/s** | AWS Open Data mirror. No credentials. Measured 2026-07-26: 133,822,024 bytes in 5.4 s = 24,997,618 B/s. |

**~250× difference.** Full 133 GB challenge-2018 corpus: 96 hours vs ~90 minutes.

```bash
# The one line that matters
aws s3 cp --no-sign-request --recursive \
  s3://physionet-open/challenge-2018/1.0.0/training/ ./c18_raw/

# Or per-file over HTTPS, no aws CLI needed
curl -O https://physionet-open.s3.amazonaws.com/challenge-2018/1.0.0/training/tr03-0005/tr03-0005.mat
```

### Why not host our own bucket

Considered and rejected. To own the raw data we would:

1. Download 133 GB from the AWS mirror (~90 min)
2. Upload 133 GB to our bucket (~90 min)
3. Pay ~$3/mo storage indefinitely
4. Pay **egress on every Chapman pull** — $0.09/GB ≈ **$12 per full download**

The mirror serves identical bytes at identical speed for $0, because AWS sponsors the Open Data program. For a frozen public archive, not owning it is a feature: it cannot be accidentally deleted, and it cannot generate a bill.

Ownership matters for *derived* artifacts, which are ~2 GB and belong in tier 2.

---

## 2. Dataset inventory

### 2.1 PhysioNet/CinC Challenge 2018 — "You Snooze You Win" (PRIMARY, in acquisition)

The dataset this session was about acquiring.

| Property | Value |
| --- | --- |
| Full name | PhysioNet/Computing in Cardiology Challenge 2018 |
| Source corpus | Massachusetts General Hospital sleep lab |
| Labelled training subjects | **994** |
| Total size | ~133 GB (~134 MB/subject) |
| Sample rate | 200 Hz, all channels |
| Channels | 13 |
| License | Open, **no application required** |
| S3 prefix | `s3://physionet-open/challenge-2018/1.0.0/training/` |
| Layout | `training/tr03-0005/tr03-0005.{mat,hea,arousal}` |

**Channels:** F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1 (EEG) · E1-M2 (EOG) · Chin1-Chin2 (EMG) · ABD, CHEST, AIRFLOW (respiratory) · SaO2 · ECG

**Why this dataset unblocks the stalled experiment:** arousals are annotated **by cause**, which is what the autonomic-precedes-cortical hypothesis requires:

- `arousal_spontaneous` — non-respiratory
- `arousal_rera` — respiratory effort related
- `arousal_obstructiveapnea`, `arousal_centralapnea`, `arousal_mixedapnea`, `arousal_hypopnea` — respiratory

Prior data (slpdb) had **190 non-respiratory arousal events across 9 subjects, 73% from a single person** — hopelessly underpowered and confounded by subject identity. Challenge-2018 is ~100× the cohort with cause labels.

**Annotation format quirk:** arousal intervals use bracket syntax in the `.arousal` annotation `aux_note` field — `(arousal_spontaneous` opens, `arousal_spontaneous)` closes. Handled by `_parse_arousal_epochs`.

### 2.2 Local datasets already on disk

At `/Users/aayu/datasets/kahlus_multidataset_public/` — **38 GB total**.

| Dataset | Size | Files | Purpose |
| --- | ---: | ---: | --- |
| `ds005620/` | 30 GB | — | OpenNeuro propofol sedation EEG. Anesthesia arm of Passive PCI (Track F1) cross-etiology test. |
| `sleep-edfx/` | 7.1 GB | 306 | Sleep-EDF Expanded. M2 gate machinery, sleep-staging baseline. |
| `slpdb/` | 387 MB | 42 | MIT-BIH Polysomnographic. The underpowered predecessor — keep for continuity checks, not for claims. |
| `nsrr/` | 8 KB | 2 | **Fingerprints only.** MESA/SHHS require credentialed access. Application not yet started. |
| `chbmit/` | 8 KB | 2 | Fingerprints only. CHB-MIT seizure — M3 gate failed honestly here (underpowered). |
| `siena-scalp-eeg/` | 4 KB | 1 | Placeholder. |
| `eegmmidb/` | 4 KB | 1 | Placeholder. |

**Policy:** raw neural data is never committed to git. Only fingerprints (record list + per-file sizes + sha256 of header set) go in the repo, so a downstream run can assert it read the same cohort.

### 2.3 NSRR (MESA / SHHS) — application not started

Credentialed access via sleepdata.org, **weeks of lead time**. Flagged as a pure latency hedge: costs nothing to have the application pending while other work proceeds. **Still not started — this is the cheapest open action on the list.**

---

## 3. Work completed this session

### 3.1 Adapter and tests (committed)

- `src/neurotwin/adapters/challenge2018.py` — streaming adapter. Arousal-interval parser, stage forward-fill, per-second autonomic grid, EEG decimation to 32 Hz (keeps sigma/spindle band resolvable).
- `tests/adapters/test_challenge2018.py` — 12 unit tests, all passing. **Caught a real bug**: `_parse_stages` divided sample index by 30 instead of 30×fs, i.e. off by a factor of `fs`.
- `scripts/fetch_challenge2018_features.py`, `scripts/fetch_challenge2018_parallel.py` — extractors with file-based claim locking so workers don't double-process.

Commits: `05366cae`, `621ce259`.

### 3.2 QRS detection — the real bottleneck (NOT committed, NOT verified)

`wfdb.xqrs_detect` takes **147 s/record** — a pure-Python sample loop over 5.1M samples. Across 994 records that is ~40 hours of CPU, and it was the actual 20-hour bottleneck, not the network.

`src/neurotwin/adapters/qrs_fast.py` — vectorized Pan-Tompkins implementation, **0.32 s vs 147 s (~460× faster)**.

Validation against xqrs on 3 records:

| Record | Beat match @100 ms | HR-series Pearson r |
| --- | ---: | ---: |
| tr03-0029 | 99.96% | 0.977 |
| tr03-0052 | 97.5% | 0.546 |
| tr03-0005 | 98.9% | **0.148** |

**Diagnosis: T-wave false positives.** 27,158 detections vs xqrs's 21,794 on tr03-0005. Extra detections split true RR intervals and shred the derived rate series even though per-beat matching looks excellent. High beat-match with low HR correlation is the signature.

A slope-based T-wave discriminator was added (360 ms refractory window, 0.55 slope ratio threshold). **Revalidation was in flight when development was paused. This code is unverified and must not be used for any claim until the three-record comparison is re-run and HR r > 0.95 across all of them.**

### 3.3 A scientific error, caught and corrected

I initially stored ECG as a **log-power envelope** (amplitude) rather than QRS-derived **heart rate** (timing) — purely to dodge the 147 s xqrs cost. A collaborator caught it. The entire hypothesis (Bonnet & Arand; Togo) concerns *rate*, not *amplitude*. Envelope cannot test it.

75 subjects processed this way were quarantined and deleted. **Recorded here because the failure mode — letting a compute constraint silently redefine the measured quantity — is exactly what the gate discipline exists to catch, and it nearly got through.**

### 3.4 Hetzner staging box

`root@178.105.177.223` — 2 vCPU, 3.7 GB RAM, 38 GB disk, Nuremberg (nbg1), venv at `/root/venv` with wfdb/numpy/scipy/neurokit2.

Current state: **219/994 subjects, 30 GB in `/root/c18_raw/`, 3.0 GB free, downloader stopped.**

`/root/c18_batch.sh` with `next` / `status` / `clear` subcommands; tracks state in `_fetched.txt` so `clear` frees disk without losing resume position.

**This box is now obsolete for data staging** — see §4. Chapman pulls from S3 faster than this box can relay.

---

## 4. The plan

### 4.1 Storage decision (resolved)

**No Hetzner volume.** It was priced at $19.18/mo for 200 GB, ext4 vs xfs. Rejected because:

- The volume is region-locked to nbg1. **Chapman cannot mount it.**
- It would stage a copy of data Chapman can fetch faster from S3 directly.
- Hetzner egress would be paid to ship it to California.

**No own S3 bucket for raw.** See §1.

**Adopted:**

```
raw 133 GB      → AWS Open Data mirror (already there, $0, immutable)
derived ~2 GB   → Hugging Face dataset repo, or Cloudflare R2 (10 GB free, zero egress)
code/manifests  → GitHub
```

Everything produced is under our control. Nothing is paid for. Nothing needs re-uploading.

### 4.2 Chapman A100 bootstrap

Chapman is a **separate cluster** — not Hetzner, not co-located. So it must be self-sufficient:

```bash
# on the Chapman node
git clone <kahlus-repo> && cd Kahlus-V1
scripts/cluster/chapman_bootstrap.sh /shared/persistent/neurotwin
```

which does:

1. Build env from `deploy/a100/environment-a100.yml`
2. `aws s3 cp --no-sign-request --recursive s3://physionet-open/challenge-2018/1.0.0/training/` (~90 min, 133 GB)
3. Feature extraction across N workers → `features/<subj>.npz`
4. Gate → JSON + Markdown artifacts
5. Push derived features + artifacts to HF/R2

Existing scaffolding to build on: `deploy/a100/`, `scripts/cluster/chapman_a100_first_run.sh` (currently just delegates to `scripts/run_full.sh`).

**Deliberate constraint: the bootstrap should fetch raw data and stop there** until the QRS detector is settled (§3.2). Running extraction with an unverified HR path would produce artifacts that fail audit.

### 4.3 Derived artifact schema

Per subject, ~2 MB compressed, ~2 GB for the full cohort:

```
features/tr03-0005.npz
  eeg_epochs           (n_epochs, 960, 2)   C3-M2 and C4-M1 SEPARATE, 32 Hz, 30 s
  hr_epochs            (n_epochs, 30)       QRS-derived bpm, 1 Hz
  resp_epochs          (n_epochs, 30, 3)    ABD, CHEST, AIRFLOW
  arousal_spontaneous  (n_epochs,)
  arousal_rera         (n_epochs,)
  arousal_respiratory  (n_epochs,)
  stages               (n_epochs,)
```

Plus, per the repo's gate contract:

- `manifest.json` — subject list, epoch counts, **and per-record which HR path fired** (xqrs / vectorized / envelope fallback), so any result can be audited for detector dependence
- split manifests — subject-held-out fold assignment, hashed
- `qrs_validation.json` — fast-detector vs xqrs per record
- RFS gate JSON + Markdown; baseline ladder tables; negative controls

---

## 5. Open items

**Blocking any scientific claim:**

1. **T-wave fix revalidation** — re-run the 3-record comparison. Requires HR-series r > 0.95 on all three before the fast detector is usable. Currently 0.148 on tr03-0005 pre-fix.
2. **QRS-derived HR primary, envelope only as fallback, and record which path each record took.** The manifest must make detector-dependence auditable. (Collaborator correction 1.)
3. **Keep C3-M2 and C4-M1 as separate channels** rather than pre-averaging. The current adapter averages them, which discards inter-hemispheric information. (Collaborator correction 2.)
4. **Confirm our epoch labels are independent of the challenge's own scoring regions** rather than assuming it. (Collaborator correction 3.)

**Infrastructure:**

5. Write `scripts/cluster/chapman_bootstrap.sh` (§4.2).
6. Decide Hugging Face vs Cloudflare R2 for derived artifacts.
7. Decommission or repurpose the Hetzner box; decide whether the 219 staged subjects are worth rsyncing or should be dropped (S3 has all 994).

**Cheap and pending:**

8. **Start the NSRR application at sleepdata.org.** Weeks of lead time, costs nothing to have running, hedges MESA/SHHS availability.

---

## 6. Claim boundary

Nothing in this document licenses a scientific claim. Per AGENTS.md, every claim still requires:

```
GatePass = split_disjoint ∧ finite ∧ beat_best_baseline
         ∧ controls_pass ∧ powered ∧ scoped_claim
```

Acquiring 994 subjects raises **power** (`C_power`: ≥8 event patients, ≥100 positive windows, ≥10–12 clusters for CI). It does nothing for the other five conditions. In particular the baseline ladder still has to be beaten — **moving average / persistence first**, since an autocorrelation exploit is the most likely explanation for any apparent arousal-forecasting signal.

The autonomic-precedes-cortical hypothesis remains **untested**, now merely testable.
