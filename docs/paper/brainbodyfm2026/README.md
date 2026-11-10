# BrainBodyFM 2026 workshop paper

Official style from
https://brainbodyfm-workshop.github.io/assets/styles/brainbodyfm-neurips-2026-style.zip
(`dblblindworkshop`, footnote: Foundation Models for the Brain and Body).

CFP: 5 pages max, excluding references and appendices. Deadline 5 Sep 2026 AoE.
Double-blind. Non-archival. OpenReview.

## What is drafted

Only `\section{Introduction}` (three paragraphs). Title is a working label.
Abstract, author block, methods, figures, tables, and references are still the
stock NeurIPS 2026 template text. Do not submit this PDF.

This PDF is current evidence only. Tests start 22 Aug 2026. The collaborator
disclaimer and build plan are kept outside the public repo.

Numbers in the intro come from the same committed artifacts as the collaborator one-pager.

## Build

```bash
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

Compiler: pdfLaTeX. No bibliography, no images, no shell-escape.

Overleaf: New Project, Upload Project, drop a zip of this directory.
