# Kahlus reading folder

Start with `01_handbook/KAHLUS_HANDBOOK.pdf`. Everything else supports it.

```
01_handbook/        the handbook (PDF + markdown source + all 5 figures)
02_cited_core/      the papers the handbook's ARGUMENT rests on  -- EMPTY, see below
03_background_eeg/  what EEG is and how the field models it      -- 6 papers, have them
04_related_work/    Coleman's lab, for the meeting               -- 1 paper, have it
```

---

## 02_cited_core — the ones to get first

These are the eight the handbook actually leans on. **We do not have any of them
locally.** Four are open access and free; four may need Stanford library access.

Read them in this order — the first two are the ones you'd be embarrassed not to
know, because the whole label-reliability argument (§3.3) is built on them.

### Tier 1 — the argument depends on these

| # | Paper | Why it matters | Access |
| --- | --- | --- | --- |
| 1 | **Lee YJ, Lee JY, Cho JH, Choi JH (2022).** Interrater reliability of sleep stage scoring: a meta-analysis. *J Clin Sleep Med* 18(1):193–202. | **Our single most important citation.** Source of κ = 0.76 overall and **N1 = 0.24**. Without it, §3.3 is an opinion. | DOI `10.5664/jcsm.9538` — free on PMC |
| 2 | **Danker-Hopfe H, et al. (2009).** Interrater reliability for sleep scoring according to the Rechtschaffen & Kales and the new AASM standard. *J Sleep Res* 18(1):74–84. | Independent confirmation, 82.0% raw epoch agreement. Two independent sources is what makes the ceiling argument hard to wave away. | DOI `10.1111/j.1365-2869.2008.00700.x` — free at Wiley |

### Tier 2 — the competition (§6)

| # | Paper | Why it matters | Access |
| --- | --- | --- | --- |
| 3 | **Kweon Y-S, et al. (2026).** SSF-SET: A Discrete EEG Token-based Framework for Sleep Stage Forecasting. *IEEE J Biomed Health Inform.* | Sleep stage forecasting on **our exact dataset**. Accuracy 0.611 / macro-F1 0.537 on SleepEDF78. **Read this one properly** — it is the state of the art we have to position against. | IEEE Xplore — needs institutional access |
| 4 | **Leguia MG, et al. (2025).** Variability and predictability in human sleep. *Brain Communications.* | NREM→REM transitions **2–4 min ahead at AUC 0.88**. The number people will quote at you. Note they used implanted sub-scalp electrodes, 8 subjects, and only one transition type. | Open access (Oxford, CC-BY) |
| 5 | **Hussain I, et al. (2022).** Quantitative Evaluation of EEG-Biomarkers for Prediction of Sleep Stages. *Sensors* 22(8):3079. | Establishes that delta power ratios track sleep depth — the physiological reason band envelopes carry state information. Classification, not forecasting. | Open access (MDPI) |

### Tier 3 — the leakage literature (§3.1 novelty claim)

The handbook claims these three all address leakage *between* examples and none
addresses leakage *inside* one example. **Verify that yourself before repeating the
claim** — it is the novelty argument for F9/F10, and if one of them does cover it,
better you find out than a reviewer.

| # | Paper | Access |
| --- | --- | --- |
| 6 | **Bernett J, et al. (2024).** Guiding questions to avoid data leakage in biological machine learning applications. *Nature Methods* 21:1444–1453. | DOI `10.1038/s41592-024-02362-y` |
| 7 | **Del Pup F, et al. (2025).** The more, the better? Evaluating the role of EEG preprocessing / CV schemes. | Search arXiv + PubMed |
| 8 | **Apicella A, et al. (2024).** Machine learning EEG data leakage. | Search arXiv + PubMed |

---

## 03_background_eeg — have these, read if terminology is the blocker

Not cited in the handbook, but if §2's vocabulary felt thin these are where it
comes from.

| File | Read it for |
| --- | --- |
| `Cohen_2017_where_does_EEG_come_from.pdf` | **Start here.** Short, readable, exactly "what is this signal actually". |
| `Buzsaki_2012_origin_of_extracellular_fields.pdf` | The definitive account of where scalp voltage comes from. Dense but canonical. |
| `Jayaram_2018_MOABB_benchmarking.pdf` | The benchmarking framework we load BNCI through. Also a good model of *how to argue a benchmark is unfair*, which is our genre. |
| `Lotte_2018_EEG_classification_review.pdf` | 10-year review of EEG classification. Good map of what the field does. |
| `Lawhern_2018_EEGNet.pdf`, `Schirrmeister_2017_deep_learning_EEG_decoding.pdf` | The two standard EEG deep-learning architectures. Context for why F8 used a small recurrent model rather than these. |

## 04_related_work

`Coleman_2025_stomach-brain_coupling_sleep.pdf` — Todd Coleman's lab, gastric–cortical
coupling during sleep. **Section 3.6** is where the directed-information argument in
your first email came from, and it is where the F9/F10 point would apply if any of
their coupling is scored over a window.

---

## Suggested order

1. `01_handbook/KAHLUS_HANDBOOK.pdf` §1, §3, §8 — twenty minutes
2. Paper **#1** (Lee 2022) — the κ table is the one number to memorise
3. Paper **#4** (Leguia 2025) — know the competition
4. `Cohen_2017` if any of §2 felt shaky
5. Paper **#3** (Kweon 2026) before writing anything for a venue

---

## A note on what is missing

The handbook cites eight papers and this folder contains none of them. That is worth
knowing rather than papering over: **the arguments in §3.3 and §6 currently rest on
abstracts and a meta-analysis table, not on full readings.** Tier 1 and #4 are free —
those four are the gap that actually matters.
