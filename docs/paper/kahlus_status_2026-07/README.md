# Kahlus status document — Coleman / Lolitkar, 25 July 2026

`kahlus_status.tex` → `kahlus_status.pdf` (9 pp). Rebuild:

```bash
pdflatex -interaction=nonstopmode kahlus_status.tex   # twice, for refs
```

Every statistic is traceable to a repository artifact, named inline. No number in
this document is estimated or rounded from memory.

| Section | Source |
| --- | --- |
| §3 calibration | `runs/autonomic_calibration/detection_power_sweep.csv` |
| §4 overlap 126x | `docs/paper/ieee_figure_source/data/amrith_overlap_headline.json` |
| §5 slpdb gate | `artifacts/slpdb_autonomic_{respiratory,non_respiratory}/` |
| §6 F2 table | `runs/amrith_isolated_check/sweep.csv` |
| §6 F4/F5 | `docs/results/findings-ledger.md` |

The document states in its opening paragraph that no positive biological finding
exists. That is accurate and deliberate: it is addressed to the senior author of
Rao et al. (2025), who reports his own coupling nulls plainly, and an inventory
that implied otherwise would not survive his reading of it.
