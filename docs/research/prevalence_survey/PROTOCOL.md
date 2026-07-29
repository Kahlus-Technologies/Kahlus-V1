# Prevalence survey — coding protocol

**Question.** In published work that forecasts EEG (or related biosignals) from its
own past, how often is the evaluation window scored with a horizon smaller than
the window length, i.e. `H < W`? By F9, any such paper reports a number inflated
by up to `W/H`.

**Why it matters.** F9 is currently a *tool*. This survey turns it into a *finding
about the field*. It is the direct analogue of McGreivy & Hakim (2024,
*Nat. Mach. Intell.*), who reviewed 76 ML-for-PDE papers and found 79% compared
against a weak baseline. That paper is the template.

**Target.** ~80 papers. Partial is fine — the survey is append-only and resumable.

---

## What to extract per paper

| field | meaning |
| --- | --- |
| `paper_id` | stable slug, `firstauthor_year_keyword` |
| `title`, `year`, `venue`, `url_or_doi` | identification |
| `task` | what is being predicted (raw signal / band power / phase / state label) |
| `fs_hz` | sampling rate, if stated |
| `W_samples` | input window length **in samples** |
| `H_samples` | forecast horizon **in samples** |
| `W_quote` | **verbatim** sentence from the paper supporting `W` |
| `H_quote` | **verbatim** sentence supporting `H` |
| `target_is_window` | `yes` if the model predicts a *window*, `no` if a single point |
| `code` | `CLEAN` / `INFLATED` / `UNRECOVERABLE` / `NOT_APPLICABLE` |
| `implied_bound` | `W/H` when `code=INFLATED`, else blank |
| `notes` | anything that made the call hard |

## Coding rules — follow exactly

1. **`NOT_APPLICABLE`** if the paper does not forecast a signal from its own past
   (classification, reconstruction, source localisation, denoising). Most hits will
   be this. Record them anyway — the denominator matters.
2. **`CLEAN`** if the target is a single future point, or a window with `H >= W`,
   or the paper explicitly states the target does not overlap the input.
3. **`INFLATED`** only if the target is a **window** and `H < W`. Record
   `implied_bound = W/H`.
4. **`UNRECOVERABLE`** if `W` or `H` cannot be determined from the text.
   **This is a legitimate and important outcome** — the rate of unrecoverable
   protocols is itself a reporting-standards result. Do not guess to avoid it.
5. **Every numeric extraction needs a verbatim quote.** If you cannot quote it,
   the code is `UNRECOVERABLE`. No exceptions, no inference from figures.
6. **Convert to samples.** If a paper says "2-second window at 250 Hz",
   `W_samples = 500`; put the original phrasing in the quote.
7. **When genuinely torn between two codes, choose the one less favourable to our
   thesis** (prefer `CLEAN` or `UNRECOVERABLE` over `INFLATED`) and say so in
   `notes`. A survey that inflates its own headline is worth nothing.

## Search strategy

Work through these in order, recording which are done in `state.json`:

1. EEG forecasting / prediction of future EEG signal
2. EEG signal prediction deep learning autoregressive
3. seizure prediction preictal window deep learning
4. brain-computer interface signal forecasting real-time
5. biosignal forecasting sliding window evaluation
6. EEG phase prediction closed-loop stimulation
7. ECG / respiration forecasting deep learning windowed

Use the Consensus MCP (`mcp__plugin_bio-research_consensus__search`, max 3 calls
per batch), PubMed, and bioRxiv. Record every paper examined, including
`NOT_APPLICABLE` ones.

## Outputs

- `survey.csv` — one row per paper, append-only. Never rewrite existing rows;
  correct by appending a row with the same `paper_id` and a `notes` entry saying
  what it supersedes.
- `state.json` — which searches have run, how many papers coded, last cycle time.
- `FINDINGS.md` — rolling summary: counts per code, the distribution of
  `implied_bound`, and any paper worth reading in full.

## What this survey may NOT do

- It may **not** write to `docs/results/findings-ledger.md`.
- It may **not** claim a prevalence rate as a project finding until a human has
  reviewed the coded rows. The CSV is evidence; the claim is a separate decision.
- It may **not** push to origin.
