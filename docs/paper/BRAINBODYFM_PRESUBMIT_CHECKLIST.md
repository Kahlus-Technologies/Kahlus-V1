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
