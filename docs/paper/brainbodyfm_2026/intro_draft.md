# BrainBodyFM 2026 — introduction draft (3 paragraphs)

Status 2026-08-20: numbers marked ⟨⟩ pin to the unified 78-fold artifact
(`artifacts/fm_probe/scores_full78_with_null.json`; see PROVENANCE.md there) and to
Krish's fine-tune 2×2 when it returns. Claims here are frozen; only the pinned
values update. P1 follows the doctor-readable framing from the 2026-08-20
advisor memo — plain task first, no capability upgrade anywhere.

---

**P1 — the task, in plain terms.**
We ask whether the next few milliseconds of a scalp EEG trace can be predicted
from the last second of that same trace, in a person the model has never seen.
This is a signal-level test, not a clinical task: nothing here diagnoses,
stages sleep, or detects seizures. Forecasting scores near r = 0.97 -- including our own, since
retracted -- suggest the answer is emphatically yes — but most of that apparent skill comes
from scoring windows that overlap the input, so the model is graded mostly on
voltage it has already been shown. When the target is restricted to strictly
future samples, the question becomes genuinely open, and the answer turns out
to be small, structured, and measurable.

**P2 — what survives honest scoring.**
Under subject-held-out evaluation with non-overlapping targets, a
regularized linear map fitted on the raw context is a far stronger baseline
than the field's evaluations acknowledge: it dominates every method we test at
80–160 ms (the 10 ms row was retired: non-causal resampling leaks future
samples into the context there; see artifacts/fm_probe/PROVENANCE.md). Foundation models pretrained on thousands of hours of EEG do not
close that gap when linearly probed — and at the one horizon where a frozen
probe does beat the linear floor (320 ms, ⟨73/78⟩ held-out subjects), an
identical architecture with random weights beats it slightly more strongly
(⟨CI excludes zero⟩): the advantage is a property of random nonlinear
features, not of pretraining. ⟨Fully fine-tuning the pretrained model at
matched budget — pending — completes this factorial.⟩ These findings extend,
to continuous forecasting, what recent benchmark audits report for
classification: tuned simple baselines remain competitive, and pretrained
representations transfer less than assumed.

**P3 — contribution.**
We contribute (i) an evaluation protocol — leakage-controlled, subject-held-
out, non-overlapping, floored by a per-fold-tuned linear model — under which
any claimed forecasting skill must be demonstrated; (ii) a measured
decomposition of short-horizon EEG predictability into a dominant linear
component, an oscillatory component the linear model subsumes, and a small
nonlinear residual (⟨10.5–17.6%⟩ MSE reduction, ⟨≥72/78⟩ subjects, ⟨1⟩ seed — cluster rerun adds seeds)
recovered by a task-trained recurrent model two orders of magnitude smaller
than current pretrained models; and (iii) the frozen/fine-tuned ×
pretrained/random factorial answering, for forecasting, the question this
workshop names directly: whether pretraining helps. On this task, at these
scales, it does not — and we release the protocol so that claim can be
falsified.

---

Notes (not part of the intro):
- Amrith's stimulation-threshold thesis is deliberately absent here — wrong
  venue. It belongs in the Amrith-facing one-pager alongside the clinical
  capability table from the memo ("this is not a diagnostic tool").
- Anonymization: no Kahlus name, no repo URL until camera-ready.
