# Version 7 edit sheet — Dr. Zaidi's call, 2026-08-03

Every item below is either something he asked for on the call, something his
typed note in the document asks for, or a defect found while doing that work.
Paste the **AFTER** text; the **FIND** text is what is in Version 6 today.

Source of the asks: the 18-minute screen share, timestamps in brackets.

---

## 1. Figure 1 — replace the image  [13:20-13:32]

> *"this picture, this figure, if you can draw quickly tonight, this size"*
> *"I'm wondering why it's so low quality. It's very blurred."*

**Do:** delete the current Figure 1 image, insert
`figures/fig1_concept.pdf` (or `.png` if Docs rejects the PDF).

**Insert at 100%. Do not drag to resize.** It is authored at 3.5 in wide, the
IEEE column measure. Scaling it down is what produced the blur he flagged.

**Caption — FIND (this literal placeholder is still in the document):**

    Figure 1. [DONE - now called in Section II.]

**AFTER:**

    Fig. 1. Kahlus-V1 concept. (A) Noninvasive EEG is acquired as a past
    window of 22 channels x 128 samples. (B) A latent neural field carries the
    inferred brain state, which is evolved causally rather than measured
    directly. (C) A readout operator emits a future window whose sample
    indices are disjoint from the input by construction. (D) An evidence gate
    releases only those claims the measured comparison supports.

---

## 2. Figure 3 — replace with the 2x2  [13:42-14:05]

> *"cut it into half from here... half and then half under it. So that will
> give us ABCD. So they can appear in one column."*

**Do:** delete the wide four-across strip, insert `figures/fig3_protocol.pdf`.
Same rule: place at 100%, single column.

**Caption — FIND:**

    Figure 3: Evidence gates and leakage controls. Subject-held-out splits prev...

**AFTER:**

    Fig. 3. Evaluation protocol and evidence gates. (A) When input and target
    windows share samples, correlation is inflated and the task is not a
    forecast; only the disjoint configuration is scored. (B) Splits are held
    out by subject, so no subject appears in both training and test; the two
    test subjects available remain below the threshold for a powered claim.
    (C) On the isolated subject-held-out task at a horizon of 16 samples, the
    recurrent model reaches MSE 0.690 against 1.334 for persistence, while
    ridge regression reaches 0.517. (D) The gate therefore releases the
    baseline-improvement claim and withholds the claim of superiority over a
    linear model.

---

## 3. In-text callouts, in red  [13:38, 14:06]

> *"call it in the main text somewhere and make the red sign there that you
> called it."*

He pointed to this existing sentence as the pattern to copy:

    The architecture of this evaluated path is shown in Figure 2, and the
    component-level configuration is listed in Table II.

**Figure 1** already has a callout in Section II ("Figure 1 presents the
overall Kahlus-V1 concept..."). Mark that sentence red so he can see it.

**Figure 3** already has one ("Figure 3 summarizes the evidence-gate logic...").
Mark it red.

**Table IV has no callout and no caption.** Add the caption above the table:

    TABLE IV. Supported and Unsupported Claims Under the Recovered Evidence

---

## 4. Numbering — the confusion he flagged  [14:48-15:01]

> *"evaluated in figure two, is in table two... we should put this as figure
> two. Or if it is figure two, I don't know what it is."*

He is right, and the cause is mechanical: **body text uses Roman numerals,
captions use Arabic.** The same table is "Table II" in the paragraph and
"Table 2." in its caption, so it reads as two different tables.

IEEE style: tables take Roman numerals, figures take Arabic. Fix the captions.

| Caption today | Change to |
|---|---|
| `Table 1. Main Model Families Relevant to the Current Work` | `TABLE I. MAIN MODEL FAMILIES RELEVANT TO THE CURRENT WORK` |
| `Table 2. Model (Kahlus V1) Configuration` | `TABLE II. MODEL (KAHLUS-V1) CONFIGURATION` |
| `Table 3. Kahlus-V1 results comparison with the baseline (Ridge Model)` | `TABLE III. RESULTS COMPARISON WITH RIDGE AND PERSISTENCE BASELINES` |
| *(missing)* | `TABLE IV. SUPPORTED AND UNSUPPORTED CLAIMS UNDER THE RECOVERED EVIDENCE` |
| `Figure 2: Kahlus v1 Neural Field Compiler architecture...` | `Fig. 2. Kahlus-V1 Neural Field Compiler architecture...` |

Also unify `Figure N:` to `Fig. N.` across all captions, per IEEE.

---

## 5. Scaffolding notes still in the manuscript

These are working notes to yourselves that would ship to a reviewer. **Delete
all three:**

- `Figure 1. [DONE - now called in Section II.]`  (replaced by the caption in item 1)
- `[DONE - Section VII written below.]`
- `[DONE - Conclusions and Discussion written below.]`

---

## 6. His typed note in the document

Sitting after reference [24]:

> *"If you want to add these figures to the results section, then please remake
> these figures and call them in the result section. Also add discussion on
> these results in the 'Discussion section'."*

Four uncalled images follow it. On the call [13:11] he said *"we don't have to
add all of them"* — so this is optional for tonight. **Recommended: delete the
trailing uncalled images.** An uncalled figure after the references is a
straightforward reviewer complaint, and the two figures above already carry the
argument. If you would rather keep one, it needs a caption, a numbered slot, a
red callout in Results, and a sentence in Discussion.

---

## 7. Consistency check on the numbers in the figures

Figure 3C plots **0.690 / 1.334 / 0.517**, which are the isolated,
subject-held-out values in Section VI. It deliberately does **not** show
MSE 3.116 or r 0.972.

That is not a stylistic choice. Section VI retracts those numbers as an
artifact of window overlap, and the old figure displayed them next to the word
"APPROVED". Shipping that would have contradicted the paper's own retraction in
the figure a reviewer reads first.

---

## Files

    docs/paper/iecbes2026/figures/
        fig1_concept.pdf     <- insert for Figure 1 (vector, preferred)
        fig1_concept.png     <- 600 dpi fallback
        fig1_concept.tiff    <- 600 dpi LZW, if IEEE asks for TIFF
        fig3_protocol.pdf    <- insert for Figure 3
        fig3_protocol.png
        fig3_protocol.tiff

Rebuild with:

    cd docs/paper/iecbes2026/src
    python3 make_fig1_concept.py && python3 make_fig3_protocol.py
    python3 check_geometry.py     # must print PASS

---

## The text to send him  [17:05]

> *"you text me that you have uploaded this picture and the other figure, and
> you have called them somewhere in the text. Then I will start reading it."*

Draft:

> Dr. Zaidi — Figure 1 is redrawn at column width and Figure 3 is now the 2x2
> ABCD layout, both uploaded to Version 7. Both are called in the main text and
> marked in red. I also matched the table numbering to the body text (Roman) so
> Table II reads consistently. Ready for your read.
