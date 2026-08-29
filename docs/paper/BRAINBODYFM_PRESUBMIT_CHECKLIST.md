# BrainBodyFM pre-submission checklist — mechanical failures only

Audited 2026-08-28 against the draft at `1c13179e`. These are the things that
get a paper desk-rejected regardless of how good it is. None of them require
the framing decision.

## Anonymization: currently CLEAN

Searched the draft for `Amrith`, `Lotlikar`, `Aayushya`, `Patel`, `Krish`,
`Garg`, `Zaidi`, `Kahlus`, `NeuroTwin`, `Cupertino`, `Stanford`, `SJSU`,
`ptlnextdoor`, `kgarg`, `github.com`, `Kahlus-Technologies`.

**Zero hits.** The author block is still the stock `David S.~Hippocampus`
placeholder, and there is no acknowledgements section. Nothing to strip today.

**But the model name is the trap.** The moment the results section is written,
"Kahlus" enters the paper as the model name and the submission is
de-anonymized — the repo is public and the name is unique. Decide now:

- refer to the model as **"our model"** or a neutral placeholder throughout, or
- pick a blind alias and use it consistently.

Every appearance in prose, table headers (`kahlus_gru`), and figure labels must
use the same choice. **The results table currently carries `kahlus_gru` as the
reference entrant name in every row** — that is the single most likely
anonymization failure in this submission.

## Style file: MUST be swapped

Active style line, L10:

```
\usepackage[dblblindworkshop]{neurips_2026}
```

`dblblindworkshop` is the **correct option** for a double-blind workshop — good.
But the CFP requires the workshop's own modified style, distributed as
`brainbodyfm-neurips-2026-style.zip` from the call-for-papers page. The draft
currently uses stock `neurips_2026`. **Download their zip and build against it**
before submitting; a stock build may differ in margins or the header.

## The F10 comment: delete before submitting

Lines still in the source:

```
% NOTE: the 223-cell / r=0.984-vs-0.014 correlation numbers are the F10
% artifact on PR #77 (finding/correlation-inflation-law); cite only once merged.
```

Two reasons to remove it:

1. It names a PR number and an internal branch, which is an identity leak in a
   double-blind submission.
2. Its condition is now satisfied — F10 merged via PR #81 — so the note is
   obsolete. **Cite the numbers, drop the comment.**

LaTeX comments do not render, but they *are* in the `.tex` if source is ever
shared, and OpenReview submissions are PDFs, so the practical risk is low — the
identity leak matters only if the source travels. Delete it anyway; it costs
nothing.

## Length: 5 pages max, excluding references and appendices

Current draft is ~1.5 pages of content plus ~375 lines of NeurIPS template
boilerplate that must be deleted. After deletion, the intro alone is ~1.5 pages,
leaving ~3.5 for methods, results, the gate-failure section, and discussion.

**Build and count pages before the final day.** A 5.2-page paper at 23:00 on
Sept 5 is a worse problem than a 4.5-page one.

## URLs already in the draft

Seven URLs, all from the template (`neurips.cc`, CTAN, TeX StackExchange). None
identify the author. They will disappear with the boilerplate.

## Submission mechanics

- **OpenReview**, group `NeurIPS.cc/2026/Workshop/BrainBodyFM`
- **Deadline Sept 5, 2026 AoE** — Anywhere on Earth, so effectively end of
  Sept 5 in the last timezone. Do not rely on that margin.
- Non-archival; concurrent NeurIPS main-conference submission is permitted
- Notification Sept 29, camera-ready Nov 6, workshop Dec 11 or 12, Sydney
- 9 submissions get spotlight talks

## Order of operations for the last day

1. `git pull` (local main is 75 commits behind)
2. build against the workshop style zip
3. run `python3 scripts/verify_paper_numbers.py` — must exit 0
4. grep the built PDF text for the name list above — must return nothing
5. count pages
6. submit **Sept 4**, not Sept 5

The one lesson this project keeps re-learning: the deadline that slips is the
one with no external reminder. Sept 4 gives a day of margin against exactly that.

---

## ACCEPTANCE PATH EXERCISED 2026-08-28 — the paper actually builds

Everything above was inspection. This is the real thing: the draft compiled to a
PDF with the generated tables in it, and the PDF was checked the way a reviewer
or the submission system would encounter it.

**What was run:** extracted `neurips_2026.tex` and `neurips_2026.sty` from commit
`1c13179e`, generated the tables with `scripts/emit_paper_tables.py`, wired them
in with `\input{tables.tex}`, and compiled with `pdflatex -halt-on-error`.

| check | result |
|---|---|
| compiles | **yes**, exit 0, no LaTeX errors |
| PDF produced | 214 KB |
| generated tables render | **yes** — both the paired-comparison table and the gate table appear in the PDF text |
| **anonymization of the rendered PDF** | **CLEAN** — zero hits across 15 identifying terms in the extracted text |
| page count, as-is | **8 pages** |
| page count, template deleted | **3 pages** |

### The page finding, which is good news

As-is the document is 8 pages against a 5-page limit — but pages 2 through 6 are
**NeurIPS template boilerplate** ("Sample figure caption", "Note that amsfonts is
automatically loaded…"). Deleting the 372 lines of template leaves:

**3 pages: the written intro plus both generated tables.**

So the real budget is **2 pages of content already written, ~3 pages free** for
methods, the gate-failure section, discussion, and limitations. That is a
comfortable fit rather than a squeeze, which was not knowable before compiling.

### What this changes

- The "count pages before the final day" warning above is now **resolved in
  advance**: there is room.
- Deleting the template is not cosmetic, it is the difference between 8 pages
  and 3. Do it first, before writing anything else.
- The anonymization check now has a real target. Re-run it on the final PDF:

```
pdftotext paper.pdf - | grep -icE 'kahlus|neurotwin|aayushya|patel|zaidi|sjsu|stanford'
```

Must return 0. It currently does, because the tables use the blind alias.

### Honest limit — RESOLVED 2026-08-28

I previously flagged that the build used the `.sty` committed with the draft
rather than the workshop's required style. **Downloaded the official
`brainbodyfm-neurips-2026-style.zip` and diffed it: the committed
`neurips_2026.sty` is BYTE-IDENTICAL to the workshop's.** The draft was already
on the right style; there is nothing to swap.

Rebuilt against the downloaded official copy to confirm, with the template
stripped and the generated tables inserted:

| check | result |
|---|---|
| compiles | exit 0 |
| **pages** | **4 of 5 allowed** |
| anonymization of rendered PDF | **CLEAN** |
| both tables render | yes |

So the real state is **4 pages used, 1 page free**, with the intro and both
tables in place. Tighter than the 3-page figure from the earlier build (which
also stripped the references section), and the honest number to plan against.

**Note on the separate `brainbodyfm2026.zip` in the repo:** that is a submission
bundle (tex + sty + README), not the style distribution. Do not confuse the two.
