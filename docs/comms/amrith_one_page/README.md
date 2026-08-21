# Kahlus one-pager for Amrith Lotlikar

`main.tex` builds standalone with `pdflatex main.tex`. No bibliography, no
shell-escape, no external images — it compiles on Overleaf free with zero setup.

## Sources of every number

| claim | artifact |
|---|---|
| 3.116 / 0.972 / 7.745, retracted | `docs/research/NEURAL_CASP_TECHNICAL_DOCUMENTATION.md` |
| 127-of-128 overlap, 126x measured | `eeg-forecastability-limits` README; 0.227/0.0018 = 126.1 |
| W/H on 286 cells, 0 violations (F9) | `docs/KAHLUS_HANDBOOK.md` §3.1 |
| correlation law, 223/286 violated (F10) | `docs/KAHLUS_HANDBOOK.md` §3.2 |
| 0.211 / 0.227 / 0.281, 0.690 / 0.822 / 1.334 | `NEURAL_CASP_TECHNICAL_DOCUMENTATION.md` F2 table |
| BNCI LOSO 0 of 9 horizons | `F8 did not replicate` |
| residual stack table + CIs | `artifacts/isolated_residual_evidence_20260815/VERDICT.md` |

## Upload to Overleaf

Overleaf has no public "create project" API on free tier, so this is manual —
30 seconds:

1. `zip -r amrith_one_page.zip main.tex README.md`
2. Overleaf → **New Project → Upload Project** → drop the zip.
3. Menu → Compiler: **pdfLaTeX** (default, already correct).

Better, if you want it to stay in sync with the repo: Overleaf → New Project →
**Import from GitHub** → `Kahlus-Technologies/Kahlus-V1`, then set the main
document to `docs/comms/amrith_one_page/main.tex`. That needs Overleaf premium
for GitHub sync; the zip path does not.

The yellow box at the top is Amrith-only working notes (tests 22 Aug 2026,
five-lever build plan). The body below the box is current evidence only.

## Before sending

- The residual section says "powered preliminary positive." Do not upgrade that
  wording until the seed sweep and the GBM baseline are done. VERDICT.md lists
  all five blockers.
- Do not put the residual numbers in the IECBES submission — one seed,
  co-author-unreviewed.
