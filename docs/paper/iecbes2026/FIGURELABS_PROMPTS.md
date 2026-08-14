# FigureLabs prompts — Kahlus-V1 / IECBES 2026

Paste one prompt per generation. They are written to match the house style
already set by `source_art/figure1.png` and `source_art/architecture.png`, so
the new panels drop in beside the existing ones without a visual seam.

---

## Two rules that decide whether this works

**1. Resolution. This is the whole reason Figure 1 got flagged.**

Dr. Zaidi's complaint ("it's very blurred") was not a FigureLabs quality
problem. It was a placement problem: art drawn for a wide slide, dragged down
to a 2-inch box in the doc. Printed detail is

    effective_dpi = image_px_wide / placed_width_in

At IEEE column width (3.5 in) you need **>= 2100 px wide for 600 dpi**. Ask
FigureLabs for the largest export it offers, and never accept a panel under
**1600 px** on its long edge. If a panel will sit half-column (1.7 in), 1200 px
is the floor.

**2. No numbers in the art.**

Every generated panel must be free of metrics. The old Figure 1 has
`MSE=3.116, Pearson r=0.972` baked into panel C, and Section VI of the paper
*retracts those values* as an artifact of window overlap. Numbers go in as a
matplotlib overlay from CSV, so they can never drift from the data again.

Where a prompt below needs a value slot, it says "leave an empty rounded box".

---

## Shared style block

Append this to every prompt so the panels match each other.

> Style: clean scientific journal illustration, flat vector look, white
> background. Thin 1.5 px strokes. Rounded-corner rectangles with pale fills
> and saturated outlines. Palette: teal #0F766E, deep blue #1E4E9C, green
> #15803D, amber #B45309, warm red #B3261E, neutral grey #5F6368, near-black
> #1A1A1A for type. Sans-serif labels (Inter or Helvetica), medium weight,
> generous size, no thin hairline type. No drop shadows except where stated.
> No gradients on boxes. No 3D bevels. No decorative icons. Nothing
> photographic. Export at the maximum resolution available.

---

## Prompt 1 — Head and electrode montage  *(replaces Fig 1 panel A)*

> A schematic top-down view of a human head for an EEG methods figure. Circular
> scalp outline in medium grey, with a small nasion bump at the top and a small
> ear shape on the left and right edges. Inside the scalp, 22 small circles
> arranged in a symmetric 10-20 montage, laid out in rows of 3, 5, 6, 5, 3.
> Each circle is white-filled with a teal outline and contains a small number,
> 1 through 22, in teal. Below the head, thin teal lines converge downward from
> the lowest electrodes into a single rounded teal-filled bar labelled "EEG
> measurement" in white text. Beneath that bar, two lines of teal text: "Past
> EEG window:" and "22 channels x 128 samples". No title, no panel letter, no
> border around the whole panel. Portrait orientation, roughly 3:4.
>
> *(+ shared style block)*

---

## Prompt 2 — Latent neural field surface  *(replaces Fig 1 panel B)*

> A smooth 3D mathematical surface for a scientific figure, drawn as a fine
> wireframe mesh in deep blue on white. The surface has two rounded peaks of
> different heights and one shallow trough, like a smooth Gaussian mixture,
> viewed from a low oblique angle. The mesh lines are thin and evenly spaced in
> both directions. Directly beneath the surface, a dashed blue ellipse
> indicating the ground plane, with no fill. Above the surface, one line of
> bold blue text: "Hidden brain-state representation". No axes, no gridlines,
> no numbers, no colour fill on the surface itself, no panel letter, no border.
> Landscape orientation, roughly 4:3.
>
> *(+ shared style block)*

---

## Prompt 3 — Architecture flowchart  *(replaces Figure 2)*

Regenerate this one because the current version is the strongest art you have
and it only needs a vertical re-cut, not a redesign.

> A vertical flowchart for a machine-learning architecture figure, white
> background, single column, five stacked stages connected by downward arrows.
>
> Stage 1: rounded rectangle, pale blue fill, navy outline. Bold heading
> "Observed EEG window", below it two lines of smaller text "[B, 127, 22]" and
> "past EEG samples".
>
> Between stage 1 and 2, no box, just five short centred lines of plain dark
> text: "Linear 22 -> 384", "Conv1d k=3", "Conv1d k=1", "GELU + residual",
> "LayerNorm(384)". Below them a bold label "Field inference" and a downward
> teal arrow.
>
> Stage 2: rounded rectangle, pale green fill, teal outline. Bold heading
> "Latent neural field", below it three lines "384-d hidden state",
> "+ modality embedding", "input-conditioned latent representation". Downward
> teal arrow.
>
> Stage 3: rounded rectangle, pale green fill, teal outline. Bold heading
> "Field dynamics", below it three lines "12-layer GRU", "hidden size 384",
> "no dropout". Downward teal arrow.
>
> Stage 4: rounded rectangle, pale amber fill, amber outline. Bold heading
> "Observation operator", below it two lines "Linear 384 -> 22" and
> "future EEG window".
>
> To the right of the whole stack, a tall narrow rounded rectangle with a
> dashed grey outline and very pale grey fill, containing bold grey text
> "Future operators", smaller grey text "not evaluated here", and below it
> three grey words stacked vertically: "fMRI", "fNIRS", "behavior".
>
> No title, no panel letter, no parameter counts, no footer text. Portrait
> orientation, roughly 2:3.
>
> *(+ shared style block)*

---

## Prompt 4 — Window overlap / leakage  *(Fig 3 panel A)*

> A schematic timeline diagram for a machine-learning evaluation figure, white
> background, showing two cases stacked vertically.
>
> Top case, labelled in bold warm red above it: "overlapping". Two long
> horizontal bars on the same baseline that partially overlap in the middle.
> The left bar has a pale blue fill and navy outline. The right bar has a pale
> red fill and red outline. The overlapping region where they cross is a
> solid, more saturated red. Below this pair, one line of red text:
> "shared samples inflate correlation".
>
> Bottom case, labelled in bold green above it: "disjoint". Two horizontal bars
> of equal length, side by side, separated by a small clean gap, not touching.
> Left bar pale blue fill, navy outline, containing the word "input" in navy.
> Right bar pale green fill, green outline, containing the word "target" in
> green.
>
> No numbers, no axis, no tick marks, no panel letter, no border. Landscape
> orientation, roughly 4:3.
>
> *(+ shared style block)*

---

## Prompt 5 — Subject-held-out split  *(Fig 3 panel B)*

> A schematic diagram of a dataset split for a machine-learning figure, white
> background. A single horizontal row of seven tall rounded rectangles of equal
> size, evenly spaced with small gaps. The first five have a pale blue fill and
> navy outline. The last two have a pale amber fill and amber outline. Each box
> contains a short subject label, S1 through S7, in the matching colour.
> Above the first five boxes, a bold navy label "train", positioned centrally
> over that group. Above the last two, a bold amber label "held out",
> positioned centrally over that group. A thin vertical dashed grey line runs
> between the fifth and sixth boxes to mark the split boundary.
>
> No numbers besides the subject labels, no panel letter, no border, no
> percentages. Landscape orientation, roughly 3:2.
>
> *(+ shared style block)*

---

## What we do NOT regenerate

These stay in matplotlib, because every value must trace back to a CSV row and
survive a reviewer asking where it came from.

| Panel | Data source |
|---|---|
| Horizon sweep, MSE vs forecast horizon | `ieee_figure_source/data/amrith_horizon_sweep.csv` |
| h=16 baseline bar chart | same file, `horizon == 16` |
| Any table | manuscript |

A generated image cannot be audited. A plot read from CSV can.

---

## How the merge works

1. You generate panels 1-5 in FigureLabs, export at max resolution.
2. Drop them in `docs/paper/iecbes2026/art/` with these names:

       art/head_montage.png
       art/latent_field.png
       art/architecture.png
       art/overlap.png
       art/subject_split.png

3. I composite: generated art for the illustrative panels, matplotlib for the
   data panels, panel letters and captions authored at 3.5 in so type prints at
   the right size.
4. `check_geometry.py` gates the merged result on type size, containment, and
   overlap, plus a new resolution check that fails the build if any placed
   panel would print under 600 dpi.

That last check is what stops a repeat of the blur complaint.
