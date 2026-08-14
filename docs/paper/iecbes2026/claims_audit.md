# Claims audit — V7.1 → V7.2 (2026-08-13)

Every numeric/scientific claim in `Paper-Aayushya-Krish-Zaidi-V7.1.docx`, checked against
repo artifacts and `docs/results/findings-ledger.md` (incl. the `raptor3/delete-unreachable`
ledger, which carries the BNCI verdict). Verdicts: KEEP (verified), REWORD (misleading or
underspecified), ADD (required honest result missing), DELETE.

## Number-by-number provenance

| Claim in V7.1 | Source artifact | Verdict |
| --- | --- | --- |
| MSE 3.116 / MAE 1.297 / r 0.972 / R² 0.942 (Kahlus), ridge 7.745/2.150/0.925/0.856, persistence 8.204/0.924/0.847, masked recon 53.977/−0.013 | `docs/results/a100-run-history.{md,csv}` | KEEP — already framed as retracted (F1). Two sentences still credit it as meaningful → REWORD (below) |
| Ridge 0.0018 overlapping vs 0.227 strictly-future, factor ≈126 | `runs/amrith_isolated_check/headline.json` (0.001796 → 0.226937, ×126.3; note says "Subject-held-out Sleep-EDF") | KEEP, but dataset is Sleep-EDF and the draft never says so → REWORD (name corpus) |
| Isolated sweep: h=16 GRU 0.690 vs persistence 1.334, mean-of-trace 1.264, ridge 0.822 | `docs/paper/ieee_figure_source/data/amrith_horizon_sweep.csv` (committed); F2 table in `docs/paper/kahlus_status_2026-07/kahlus_status.tex` captioned **subject-held-out Sleep-EDF @100 Hz** | KEEP numbers; REWORD — draft implies BNCI (its only described dataset). h=16 = 160 ms at 100 Hz, not 64 ms at 250 Hz |
| BNCI h=1 single-channel: ridge 0.517, GRU 0.568, subject-cluster bootstrap CI [−0.086, −0.016] | `runs/bnci_isolated_check/headline.json` (0.51663 / 0.56791, CI [−0.08641, −0.01615], test subjects {8,9}) — artifact uncommitted (`runs/` gitignored); generator `scripts/bnci_isolated_forecast_check.py` committed | KEEP — negative self-claim, consistent with ledger F2. Provenance caveat noted here + CALL-NOTES |
| Split counts 12,096 / 2,016 / 4,032, 18,144 windows, 2,592 events | recovered-run bundles (a100 history) | KEEP |
| BNCI acquisition details (22 ch, 250 Hz, 288 trials/session, cue timing) | BNCI2014_001 standard description | KEEP |

## Reworded (misleading in V7.1)

1. Abstract: "These results demonstrate strong EEG future-window forecasting performance." → face-value framing; the next sentence retracts it. Overclaim removed.
2. Abstract: raw GitHub URL in body → moved to new reference [25] (Zaidi rule: no raw URLs in text).
3. Results lead-in "Kahlus-V1 improves future-window forecasting MSE and correlation relative to ridge" and "This should be interpreted as a forecasting result" → both contradict the paper's own Section VII retraction. Rewritten to point at the retraction.
4. Table III row 1 interpretation "Supported narrow forecasting result" → "Retracted: overlap-dominated (Section VII)". (Same defect class as AGENTS.md:194.)
5. Isolated-evaluation prose and Fig. 3 caption now name the corpus (Sleep-EDF, single channel, 100 Hz) — previously dataset-silent inside a BNCI-scoped paper.
6. Title/keywords/method: "Neural Field Compiler" branding removed or reduced to the hedged latent-state framing (AGENTS.md §12 legacy scope; CLAIM_EVIDENCE_MATRIX C004: the recovered result came from the GRU translator, not NFC).
7. First-person ("We …") converted to passive where clean (Zaidi 8 Aug rule). Remaining instances counted in STATUS.

## Added (required by mission §5.3, backed by ledger)

1. **Overlap inflation law (F9):** inflation ≤ W/H, verified on 286 configurations, 0 violations; predicts ×127 for W=128, H=1 vs ×126 measured. Artifacts: `artifacts/overlap_law/`, `docs/results/overlap_inflation_law_2026-07-28.md`.
2. **BNCI non-replication (ledger F26):** 9-subject LOSO × 9 horizons = 81/81 fold-horizons, per-horizon-tuned ridge, matched budget: 0/9 horizons p<0.05, 0/9 CIs excluding zero, largest gap 0.0181 MSE — corpus cannot distinguish GRU from tuned ridge at any horizon. Artifacts gitignored (`runs/bnci_crossover/`), regenerate via `scripts/bnci_crossover_sweep.py`; ledger row on `raptor3/delete-unreachable`.

## Deleted

- Trailing reviewer instructions after references ("If you want to add these figures…", "Also add discussion…") — working notes, not manuscript content.

## Not changed (deliberate)

- The Sleep-EDF F8 crossover positive (GRU beats tuned ridge <160 ms, 27 subjects) is **not** added as a headline: it is a different corpus and protocol from this paper's isolated sweep, and adding a new positive claim the night before the co-author call is a discussion item, not a unilateral edit. Listed in CALL-NOTES.
- Duplicate section number (two "VII") fixed → Conclusions becomes VIII.
