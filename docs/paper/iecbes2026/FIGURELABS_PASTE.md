# FigureLabs prompts — 5 panels, ready to paste

Each block is self-contained: context, what the panel must communicate, exact
layout, exact label text, and the style rules. Copy one whole block, paste it,
save the result under the filename in the heading into `art/`.

**Why the prompts are this long.** FigureLabs knows nothing about this project.
It renders from a written spec, so anything unstated becomes its guess. The
existing art proves it renders type from code rather than diffusing it, since
every glyph is correctly formed at 12 px, so exact label text is worth giving
and will be reproduced literally.

---

## 1. EEG acquisition (head + montage)

Save as **`head_montage.png`**  (Figure 1, panel A)

```text
CONTEXT (you have no other information about this project, so read this first):
This is a panel for a peer-reviewed biomedical engineering paper submitted to IEEE
IECBES 2026. The paper asks whether a short window of past EEG (scalp
electroencephalography) can predict the immediately following window of EEG. The
model is called Kahlus-V1. It treats measured EEG as an observation of a hidden
"latent neural field" that it evolves forward in time, then reads out a predicted
future EEG window. The paper's honest finding is that the model beats trivial
baselines but does not clearly beat linear ridge regression, and the paper is
explicit about that limit. Nothing in this panel should imply a stronger result.

AUDIENCE: engineers and clinicians reading a two-column IEEE paper in print.
The panel will be printed about 1.7 inches wide, so every label must stay legible
when reduced. Prefer fewer, larger labels over many small ones.

WHAT THIS PANEL MUST COMMUNICATE:
That the input to the model is ordinary, noninvasive, scalp-surface EEG, recorded
from many electrodes at once, and that a short slice of it forms one input window.

LAYOUT:
Portrait, roughly 3 wide by 4 tall.

Upper two thirds: a schematic top-down view of a human head, drawn as a clean
circular scalp outline in neutral grey, with a small nasion bump at the top edge
and a small ear shape on the left and right edges, so orientation is obvious.
Inside the scalp, 22 small circular electrodes in a symmetric 10-20 arrangement,
laid out in five rows containing 3, 5, 6, 5, and 3 electrodes. Each electrode is
a white-filled circle with a teal outline, containing its number, 1 through 22,
in teal. The electrodes must sit fully inside the scalp outline, none touching or
crossing it.

Lower third: thin teal lines descend from the lowest electrodes and converge into
a single rounded rectangle filled solid teal, containing the words
"EEG measurement" in white. Directly beneath that box, two centred lines of teal
text, exactly:
    Past EEG window:
    22 channels x 128 samples

HARD REQUIREMENTS:
- Output a single flat illustration on a pure white background, no page framing,
  no caption, no panel letter (A/B/C are added later by the typesetting script).
- No numbers, no metrics, no statistics, no percentages, no axis values anywhere.
- Sans-serif type only (Inter or Helvetica), medium or bold weight, no hairline
  or light weights, no italic, no serif, no condensed faces.
- Flat vector styling: 1.5 px strokes, rounded-corner rectangles, pale fills with
  saturated outlines of the same hue. No gradients, no drop shadows, no 3D bevels,
  no glow, no texture, no photographic elements, no stock icons, no emoji.
- Palette, use exactly these: teal #0F766E, deep blue #1E4E9C, green #15803D,
  amber #B45309, warm red #B3261E, neutral grey #5F6368, text near-black #1A1A1A.
- Spell every label exactly as written below. Do not paraphrase, translate,
  abbreviate, or add labels that are not listed.
- Export at the maximum resolution available, at least 2100 px on the long edge.
  This is a print figure; anything smaller reproduces blurred.
```

---

## 2. Latent neural field

Save as **`latent_field.png`**  (Figure 1, panel B)

```text
CONTEXT (you have no other information about this project, so read this first):
This is a panel for a peer-reviewed biomedical engineering paper submitted to IEEE
IECBES 2026. The paper asks whether a short window of past EEG (scalp
electroencephalography) can predict the immediately following window of EEG. The
model is called Kahlus-V1. It treats measured EEG as an observation of a hidden
"latent neural field" that it evolves forward in time, then reads out a predicted
future EEG window. The paper's honest finding is that the model beats trivial
baselines but does not clearly beat linear ridge regression, and the paper is
explicit about that limit. Nothing in this panel should imply a stronger result.

AUDIENCE: engineers and clinicians reading a two-column IEEE paper in print.
The panel will be printed about 1.7 inches wide, so every label must stay legible
when reduced. Prefer fewer, larger labels over many small ones.

WHAT THIS PANEL MUST COMMUNICATE:
That the model's internal state is a smooth continuous field which is inferred
rather than measured. It is a concept illustration, not a data plot, so it must
not look like a chart with real values.

LAYOUT:
Landscape, roughly 4 wide by 3 tall.

Centre: a smooth three-dimensional surface drawn as a fine wireframe mesh in deep
blue on white, seen from a low oblique angle. The surface has two rounded peaks of
different heights and one shallow trough between them, like a smooth mixture of
gaussian bumps. Mesh lines are thin, evenly spaced, and run in both directions
across the surface. The surface itself is unfilled, so the mesh reads as
transparent.

Below the surface: a dashed deep-blue ellipse with no fill, sitting flat like a
ground plane beneath the peaks, indicating that the surface floats above a domain.

Above the surface: one centred line of bold deep-blue text, exactly:
    Hidden brain-state representation

Below the ellipse: one centred line of grey text, exactly:
    inferred from observed signals, not directly measurable

No axes, no gridlines, no tick marks, no colour bar, no numbers.

HARD REQUIREMENTS:
- Output a single flat illustration on a pure white background, no page framing,
  no caption, no panel letter (A/B/C are added later by the typesetting script).
- No numbers, no metrics, no statistics, no percentages, no axis values anywhere.
- Sans-serif type only (Inter or Helvetica), medium or bold weight, no hairline
  or light weights, no italic, no serif, no condensed faces.
- Flat vector styling: 1.5 px strokes, rounded-corner rectangles, pale fills with
  saturated outlines of the same hue. No gradients, no drop shadows, no 3D bevels,
  no glow, no texture, no photographic elements, no stock icons, no emoji.
- Palette, use exactly these: teal #0F766E, deep blue #1E4E9C, green #15803D,
  amber #B45309, warm red #B3261E, neutral grey #5F6368, text near-black #1A1A1A.
- Spell every label exactly as written below. Do not paraphrase, translate,
  abbreviate, or add labels that are not listed.
- Export at the maximum resolution available, at least 2100 px on the long edge.
  This is a print figure; anything smaller reproduces blurred.
```

---

## 3. Model architecture flowchart

Save as **`architecture.png`**  (Figure 2 (whole figure))

```text
CONTEXT (you have no other information about this project, so read this first):
This is a panel for a peer-reviewed biomedical engineering paper submitted to IEEE
IECBES 2026. The paper asks whether a short window of past EEG (scalp
electroencephalography) can predict the immediately following window of EEG. The
model is called Kahlus-V1. It treats measured EEG as an observation of a hidden
"latent neural field" that it evolves forward in time, then reads out a predicted
future EEG window. The paper's honest finding is that the model beats trivial
baselines but does not clearly beat linear ridge regression, and the paper is
explicit about that limit. Nothing in this panel should imply a stronger result.

AUDIENCE: engineers and clinicians reading a two-column IEEE paper in print.
The panel will be printed about 1.7 inches wide, so every label must stay legible
when reduced. Prefer fewer, larger labels over many small ones.

WHAT THIS PANEL MUST COMMUNICATE:
The path a signal takes through the model: an observed EEG window is encoded into
a latent field, the field is evolved forward in time, and an observation operator
reads out the predicted future EEG window. A separate, greyed-out region shows
capabilities that exist in the framework but were NOT evaluated in this paper.
That greyed region is important; it is how the paper avoids overclaiming.

LAYOUT:
Portrait, roughly 2 wide by 3 tall. A single main column on the left, and one
narrow secondary column on the right.

MAIN COLUMN, four boxes stacked top to bottom, connected by downward teal arrows:

Box 1, rounded rectangle, pale blue fill, navy outline:
    bold: Observed EEG window
    below: [B, 127, 22]
    below: past EEG samples

Between box 1 and box 2, no box, just five short centred lines of plain dark text:
    Linear 22 -> 384
    Conv1d k=3
    Conv1d k=1
    GELU + residual
    LayerNorm(384)
then one bold dark line reading: Field inference

Box 2, rounded rectangle, pale green fill, teal outline:
    bold: Latent neural field
    below: 384-d hidden state
    below: + modality embedding
    below: input-conditioned latent representation

Box 3, rounded rectangle, pale green fill, teal outline:
    bold: Field dynamics
    below: 12-layer GRU
    below: hidden size 384
    below: no dropout

Box 4, rounded rectangle, pale amber fill, amber outline:
    bold: Observation operator
    below: Linear 384 -> 22
    below: future EEG window

SECONDARY COLUMN, to the right, spanning roughly the full height of the stack:
one tall narrow rounded rectangle with a dashed grey outline and very pale grey
fill, containing, all in grey:
    bold: Future operators
    smaller: not evaluated here
    then three words stacked vertically: fMRI, fNIRS, behavior

This right-hand column must read as clearly inactive compared with the main
column: no colour, no solid border, no arrows connecting it to the main path.

HARD REQUIREMENTS:
- Output a single flat illustration on a pure white background, no page framing,
  no caption, no panel letter (A/B/C are added later by the typesetting script).
- No numbers, no metrics, no statistics, no percentages, no axis values anywhere.
- Sans-serif type only (Inter or Helvetica), medium or bold weight, no hairline
  or light weights, no italic, no serif, no condensed faces.
- Flat vector styling: 1.5 px strokes, rounded-corner rectangles, pale fills with
  saturated outlines of the same hue. No gradients, no drop shadows, no 3D bevels,
  no glow, no texture, no photographic elements, no stock icons, no emoji.
- Palette, use exactly these: teal #0F766E, deep blue #1E4E9C, green #15803D,
  amber #B45309, warm red #B3261E, neutral grey #5F6368, text near-black #1A1A1A.
- Spell every label exactly as written below. Do not paraphrase, translate,
  abbreviate, or add labels that are not listed.
- Export at the maximum resolution available, at least 2100 px on the long edge.
  This is a print figure; anything smaller reproduces blurred.
```

---

## 4. Window overlap and leakage

Save as **`overlap.png`**  (Figure 3, panel A)

```text
CONTEXT (you have no other information about this project, so read this first):
This is a panel for a peer-reviewed biomedical engineering paper submitted to IEEE
IECBES 2026. The paper asks whether a short window of past EEG (scalp
electroencephalography) can predict the immediately following window of EEG. The
model is called Kahlus-V1. It treats measured EEG as an observation of a hidden
"latent neural field" that it evolves forward in time, then reads out a predicted
future EEG window. The paper's honest finding is that the model beats trivial
baselines but does not clearly beat linear ridge regression, and the paper is
explicit about that limit. Nothing in this panel should imply a stronger result.

AUDIENCE: engineers and clinicians reading a two-column IEEE paper in print.
The panel will be printed about 1.7 inches wide, so every label must stay legible
when reduced. Prefer fewer, larger labels over many small ones.

WHAT THIS PANEL MUST COMMUNICATE:
The central methodological point of the paper. If the input window and the
prediction target share any samples, the task is not really forecasting and the
measured correlation is inflated. Only the fully separated arrangement is a valid
test. The reader should grasp the difference instantly from the geometry alone.

LAYOUT:
Landscape, roughly 4 wide by 3 tall. Two cases stacked vertically, the invalid one
on top and the valid one below.

TOP CASE, the invalid arrangement:
One bold warm-red label above it, exactly: overlapping
Two long horizontal bars on the same baseline, which physically overlap through
their middle. The left bar has a pale blue fill and navy outline. The right bar
has a pale red fill and red outline. The region where the two bars cross is filled
a solid, more saturated red, so the shared portion is unmistakable.
One line of red text beneath, exactly: shared samples inflate correlation

BOTTOM CASE, the valid arrangement:
One bold green label above it, exactly: disjoint
Two horizontal bars of equal length side by side on one baseline, separated by a
clean visible gap, definitely not touching. The left bar has a pale blue fill and
navy outline and contains the word "input" in navy. The right bar has a pale green
fill and green outline and contains the word "target" in green.
One line of grey text beneath, exactly: strictly future samples only

No axis, no tick marks, no time labels, no numbers.

HARD REQUIREMENTS:
- Output a single flat illustration on a pure white background, no page framing,
  no caption, no panel letter (A/B/C are added later by the typesetting script).
- No numbers, no metrics, no statistics, no percentages, no axis values anywhere.
- Sans-serif type only (Inter or Helvetica), medium or bold weight, no hairline
  or light weights, no italic, no serif, no condensed faces.
- Flat vector styling: 1.5 px strokes, rounded-corner rectangles, pale fills with
  saturated outlines of the same hue. No gradients, no drop shadows, no 3D bevels,
  no glow, no texture, no photographic elements, no stock icons, no emoji.
- Palette, use exactly these: teal #0F766E, deep blue #1E4E9C, green #15803D,
  amber #B45309, warm red #B3261E, neutral grey #5F6368, text near-black #1A1A1A.
- Spell every label exactly as written below. Do not paraphrase, translate,
  abbreviate, or add labels that are not listed.
- Export at the maximum resolution available, at least 2100 px on the long edge.
  This is a print figure; anything smaller reproduces blurred.
```

---

## 5. Subject-held-out split

Save as **`subject_split.png`**  (Figure 3, panel B)

```text
CONTEXT (you have no other information about this project, so read this first):
This is a panel for a peer-reviewed biomedical engineering paper submitted to IEEE
IECBES 2026. The paper asks whether a short window of past EEG (scalp
electroencephalography) can predict the immediately following window of EEG. The
model is called Kahlus-V1. It treats measured EEG as an observation of a hidden
"latent neural field" that it evolves forward in time, then reads out a predicted
future EEG window. The paper's honest finding is that the model beats trivial
baselines but does not clearly beat linear ridge regression, and the paper is
explicit about that limit. Nothing in this panel should imply a stronger result.

AUDIENCE: engineers and clinicians reading a two-column IEEE paper in print.
The panel will be printed about 1.7 inches wide, so every label must stay legible
when reduced. Prefer fewer, larger labels over many small ones.

WHAT THIS PANEL MUST COMMUNICATE:
That the model is tested on people it never trained on. The unit of separation is
the human subject, not the individual data window. This is what makes the
evaluation conservative.

LAYOUT:
Landscape, roughly 3 wide by 2 tall.

One horizontal row of seven tall rounded rectangles of equal size, evenly spaced
with small gaps between them.
The first five have a pale blue fill and navy outline, and contain the labels
S1, S2, S3, S4, S5 in navy.
The last two have a pale amber fill and amber outline, and contain the labels
S6, S7 in amber.

Above the first five, centred over that group, one bold navy label, exactly: train
Above the last two, centred over that group, one bold amber label, exactly: held out

Between the fifth and sixth boxes, a thin vertical dashed grey line running the
full height of the row, marking the boundary.

Beneath the whole row, one centred line of grey text, exactly:
    no subject appears on both sides

No numbers other than the subject labels, no percentages, no counts.

HARD REQUIREMENTS:
- Output a single flat illustration on a pure white background, no page framing,
  no caption, no panel letter (A/B/C are added later by the typesetting script).
- No numbers, no metrics, no statistics, no percentages, no axis values anywhere.
- Sans-serif type only (Inter or Helvetica), medium or bold weight, no hairline
  or light weights, no italic, no serif, no condensed faces.
- Flat vector styling: 1.5 px strokes, rounded-corner rectangles, pale fills with
  saturated outlines of the same hue. No gradients, no drop shadows, no 3D bevels,
  no glow, no texture, no photographic elements, no stock icons, no emoji.
- Palette, use exactly these: teal #0F766E, deep blue #1E4E9C, green #15803D,
  amber #B45309, warm red #B3261E, neutral grey #5F6368, text near-black #1A1A1A.
- Spell every label exactly as written below. Do not paraphrase, translate,
  abbreviate, or add labels that are not listed.
- Export at the maximum resolution available, at least 2100 px on the long edge.
  This is a print figure; anything smaller reproduces blurred.
```

---

## After each generation

    cd docs/paper/iecbes2026/src
    python3 compose_figures.py && python3 check_geometry.py

Safe to run with only some panels present; it reports what it is still waiting
on. If a panel came back too small to print at 600 dpi it is named explicitly,
with the pixel width needed.

## If a result comes back wrong

| Symptom | Add to the prompt |
|---|---|
| Labels reworded or missing | "Reproduce every quoted label verbatim. Do not paraphrase." |
| Electrodes outside the head | "All 22 electrodes must sit strictly inside the scalp outline." |
| Looks like a photo or 3D render | "Flat 2D vector diagram only, as in a journal methods figure." |
| Type too small when shrunk | "Assume final printed width of 1.7 inches; size all type accordingly." |
| Invented extra panels or legend | "Draw only what is specified. Add no legend, caption, or extra panels." |
