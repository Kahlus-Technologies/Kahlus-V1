# Results figures (IEEE column width)

Four figures built directly from committed artifact JSON, authored at the width
they are placed at.

| figure | claim | artifact |
| --- | --- | --- |
| `fig_crossover` | the GRU beats a tuned ridge below ~200 ms and loses above it, on 27 held-out subjects | `forecast_crossover/forecast_scaleup_results.json` |
| `fig_envelope` | band-envelope forecasting keeps skill out to 60 s, and survives removing EMG and EOG | `envelope_sleepedf/results.json` |
| `fig_falsification` | the short-horizon win survives nulls, non-overlapping windows, resampled splits, and a stronger baseline | `forecast_crossover/adversarial_validation.json` |
| `fig_overlap_law` | the retracted result is explained exactly by an overlap bound of `W/H`, tested on 286 configurations | `overlap_law/overlap_law_results.json` |

## Why the figures are authored at 3.25 in

Printed type size is

    printed_pt = authored_pt * (placement_width / saved_width)

so a figure authored at 16.5 in and dropped into a 3.25 in column prints 8 pt
type at 1.6 pt. Cropping and re-stacking cannot fix it; only authoring at the
placement width can. `common.py` enforces `saved_width == 3.25 in` at save time
and refuses to write a figure that would be rescaled on the page.

This is also why `savefig.bbox` is not `"tight"`: a tight bbox grows the canvas
past `figsize` to fit labels, reintroducing the same defect at 0.8x. Constrained
layout fits labels by shrinking the axes inside a fixed canvas instead.

## Rebuilding and verifying

```
python3 src/fig_crossover.py      # and fig_envelope, fig_falsification, fig_overlap_law
python3 src/verify_figures.py     # 29/29 checks
python3 src/verify_selftest.py    # 13/13; proves each check can fail
```

`verify_figures.py` runs eight checks per figure:

1. **NUMBERS** every value in the manifest is re-read from its artifact at check
   time. Derived quantities (horizons in ms, skill against persistence, the
   synthetic-cell count) are recomputed independently of the figure code, so a
   hand-typed or stale number fails rather than passing on a loose tolerance.
2. **LEGIBILITY** the PDF's own `Tf` font records are read and the smallest type
   is asserted at or above 6.5 pt at placement width. Log-decade mathtext is the
   usual offender: a superscript is drawn at 0.7x, so a 7 pt tick prints its
   exponent at 4.9 pt.
3. **WIDTH** saved width must equal the column width.
4. **DPI** raster must be at least 300 dpi at placement.
5. **RETRACTION** the retracted values must not appear as live results. Only
   `fig_overlap_law` may name them, and only because its subject is the
   retraction; that exemption is declared, not inferred.
6. **PROVENANCE** each artifact's sha256 is recorded in `figures/provenance.json`
   and rechecked on every run, so an artifact that changes under a figure is
   caught.
7. **NO-REGRESS** the five committed hybrid figures must be unmodified, per git.
8. **DATASET** a figure may not claim a dataset its artifact does not support.
   Evidence is read from inside the artifact, not just its path.

`verify_selftest.py` plants one fault per check and asserts the check reports it.
Two checks were unfalsifiable when it was first run: NO-REGRESS passed when git
could not vouch for the files at all, and DATASET only looked at the artifact
path. Both are fixed and both now fail as intended.

## Known gap

Every figure here draws on Sleep-EDF. See `../../results/dataset_framing_gap.md`.
