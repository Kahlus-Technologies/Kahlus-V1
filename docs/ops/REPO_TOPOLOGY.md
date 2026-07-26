# Repo topology and public presence

Decision record, 2026-07-26. Supersedes ad hoc handoff notes.

---

## The decision

**Three artifacts. Not "a bunch of repos."**

| Artifact | Visibility | Holds | Maintained by |
| --- | --- | --- | --- |
| `Kahlus-V1` | **private** | research code, tests, adapters, gates | edited directly — source of truth |
| `kahlus-evidence` | **public** | gate JSONs, manifests, figures, claims ledger, withdrawal registry | **generated, never edited** |
| HF dataset repo | public | ~2 GB derived feature arrays | generated |

Raw data (133 GB) is never hosted by us — it stays on the AWS Open Data mirror,
which is free, immutable, and cannot generate a bill.

The website pulls from `kahlus-evidence` at build time. **Nothing is ever
hand-placed.**

### Why not a separate "citable" repo

`kahlus-evidence` *is* the citation target. Tag it, mint a Zenodo DOI on the tag,
cite that. A fourth repo whose only job is to be citable is a fourth thing to
forget about.

---

## The rule that makes this work

> The public repo is a **lossy one-way mirror**, produced by
> `scripts/ops/export_evidence.py`. It is never edited directly.

Hand-curating a sibling public repo means the two diverge silently and every
release becomes a reconciliation exercise. Generating it means the private repo
stays the single source of truth and the public one cannot drift.

The export is **allowlist-driven** — a file is published only if it matches a
rule, so exposing a new artifact type is a deliberate act rather than an
accident. It refuses to run from a dirty tree, because a bundle that cannot name
a real source commit is not evidence.

Verify before publishing:

```bash
python3 scripts/ops/export_evidence.py --out ../kahlus-evidence --check
```

---

## Two retention tiers

Retention is a promise, and the promise differs by artifact.

**custody** — manifests, hashes, `claim_scope`, gate results, withdrawal
registry. Tiny (~0.3 MB today). Permanent. Never rewritten. This is what a
citation resolves to, and it must still resolve in twenty years, after a dead
grant and a lost laptop.

**convenience** — figures, derived arrays. Re-derivable from custody plus public
raw data. Explicitly disposable and garbage-collectable.

Declaring the tier is declaring what you promise to keep. Publishing a 2 GB blob
with no path back to the raw archive is not evidence, so the derived-feature
bundle ships content-addressed alongside a script that re-derives one shard from
AWS and diffs the hash — 2 GB stays checkable without hosting 133 GB.

---

## Naming and citation

`Kahlus` currently means two unrelated things: a plasma robotics company with the
existing website and search presence, and this research project. Left alone, every
search collapses them and a reviewer cannot tell what they are citing.

- Research work is published as **Neural-CASP**, with Kahlus as the organisation.
- The robotics site carries a permanent, pinned disambiguation page.
- Citations point at the `kahlus-evidence` DOI, not at a repo URL that may move.

---

## Cluster dispatch

The founder cannot SSH to the A100 cluster. That constraint is fine, because
**anything inside a network can reach out even when nothing can reach in.**

```
dispatch  ->  private repo (GitHub)
              ↓ self-hosted runner polls out, no inbound ports
          collaborator's Pi
              ↓ his SSH key, never leaves his hardware
          Chapman A100  ->  evidence bundle  ->  artifact back
```

Self-hosted runners are only safe on a **private** repo — on a public one, any
stranger can open a PR that executes code on the runner. `Kahlus-V1` was public
until 2026-07-26 and is now private; it had 0 forks, 0 stars, 0 watchers at the
time of the flip.

Jobs are **named, not scripted**: the requester picks a job from `ops/jobs.yaml`
and supplies typed parameters; `scripts/ops/run_job.py` builds the command line
itself and refuses anything else. The operator audits the registry once instead
of reviewing every request, and a compromised token on the requesting side cannot
run arbitrary code on the cluster.

---

## What still needs building

Ordered by how much a missing piece costs later.

1. **Claims ledger.** `claims.yaml` keying every public number to the gate JSON
   that licenses it, with CI refusing to render a claim whose gate is missing,
   stale, or withdrawn. Retraction then propagates to the website automatically
   instead of relying on the author remembering every place a number appeared.
   The live overlap bug is the proof this is needed: it persists because nothing
   mechanically objects to it.
2. **Withdrawal registry read by CI**, so any test, figure or doc touching an
   invalidated result ID fails loudly with the retraction note in the message.
3. **Dead-man's switch** on the cluster path — the Pi posts a heartbeat, and CI
   opens an issue if nothing lands for N days. Absence of results is otherwise
   indistinguishable from absence of work.
4. **Rehydration test** — cold clone on a clean machine using only public
   artifacts, reporting time-to-first-figure. Makes undocumented environment
   knowledge visible as a number that degrades rather than a surprise discovered
   after six weeks away.
5. **Zenodo DOI** on the first `kahlus-evidence` tag.

## Deliberately not doing

- **A monorepo.** Public and private have different retention, different
  audiences, and different failure modes.
- **Hosting raw data.** ~$12 of egress per full pull to re-serve bytes AWS
  already serves free, and a bucket we own is a bucket we can accidentally
  delete.
- **Market instruments** (falsification bonds, leakage reinsurance, a public
  short register against named labs). High novelty, no near-term viability at
  this headcount, and the short register in particular would poison the
  collaborations the work depends on.
