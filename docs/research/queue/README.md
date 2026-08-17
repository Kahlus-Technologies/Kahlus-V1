# Research queue

Where the autonomous research loop puts work for a human to adjudicate.

**Nothing in this directory is a finding.** A queue item is a proposal with
evidence attached. It becomes a finding only when a person reads it, agrees, and
writes a row in `docs/results/findings-ledger.md`. That promotion step is
deliberately manual — it is the check that caught two real errors on
2026-07-28 (a baseline-tuning artifact that reversed a conclusion, and an
"evidence tracked" commit that tracked nothing).

## Item format

One file per item, `NNN-short-slug.md`:

```markdown
# <question in one line>

- **Status:** proposed | experiment-run | needs-human
- **Cycle:** <when the loop produced this>
- **Branch/commit:** <where the work lives>

## Question
What was asked, and why it matters to the project.

## What was done
Method, data, code path. Enough for someone to re-run it.

## Result
Numbers. Include the controls. If a control failed, say so first.

## What would falsify this
The specific outcome that would kill the claim. If this is blank, the item is
not ready for a human.

## Open problems
What the loop could not resolve, and what it would need.
```

## Rules for the loop

- Write items here; never edit `docs/results/findings-ledger.md`.
- Never `git push`.
- Never mark an item as a finding. `needs-human` is the terminal status.
- A result with no control is not a result — say so in the item rather than
  reporting the number alone.
- Prefer reporting a negative or a null clearly over reaching for a positive.
  Seven of this project's nine findings are negatives; that is the expected shape.
