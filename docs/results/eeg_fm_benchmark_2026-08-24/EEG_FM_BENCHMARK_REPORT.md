# External Pretrained EEG Models vs Kahlus Incumbents — Sleep-EDF Forecasting

- Schema: `kahlus.eeg_fm_benchmark.v1`
- Created: 2026-08-23T22:25:09.277468+00:00
- Mode: `merged` from 4 shard run(s) — see the Shards section
- Commit: `04818917d335b3484e5592761d91802e98762b3f` (clean worktree: True)
- Dataset: `sleep-edf-expanded-sleep-cassette`
- Normalisation: `causal_prefix` (transductive: False)
- Claim scope: `external_pretrained_eeg_baseline_ranking_sleep_edf_forecast` — scientific_claim_allowed = **False**

> This table ranks entrants on one corpus, one task, one lane. It says nothing about
> these models' published capabilities, nothing about SOTA, and nothing clinical.
> The external models are evaluated **off-task**: they are masked-infilling encoders,
> not forecasters, so a poor number is evidence about transfer, not about the model.

## Gate failures

- label-shuffle control did not collapse for every ranked entrant by the preregistered 20% margin; no passing row at a claim-eligible horizon for: biot_prest16__frozen, biot_prest16__full_ft
- time-shift control did not collapse for every ranked entrant by the preregistered 20% margin; no passing row at a claim-eligible horizon for: biot_prest16__frozen, biot_prest16__full_ft
- pretraining bought nothing on this task for biot_prest16:frozen@h1, biot_prest16:frozen@h128, biot_prest16:frozen@h2, biot_prest16:frozen@h256, biot_prest16:frozen@h32, biot_prest16:frozen@h4, biot_prest16:frozen@h64, biot_prest16:full_ft@h1, biot_prest16:full_ft@h128, biot_prest16:full_ft@h2, biot_prest16:full_ft@h256, biot_prest16:full_ft@h32, biot_prest16:full_ft@h64, cbramod:frozen@h32, eegpt:frozen@h256, eegpt:full_ft@h1, eegpt:full_ft@h128, eegpt:full_ft@h2, eegpt:full_ft@h256, eegpt:full_ft@h4, eegpt:full_ft@h64, eegpt:full_ft@h8, labram_base:full_ft@h128, labram_base:full_ft@h2, labram_base:full_ft@h64 (random-init twin at least as good as the pretrained model): plan section 5 kill criterion, so the ranking claim is blocked, not merely annotated
- control not implemented: cluster_permutation_null
- control not implemented: subject_identity_probe
- a selected hyperparameter sits on a grid edge; extend the grid and re-run
- incumbents do not beat the best external entrant at every claim-eligible horizon on the Holm-corrected paired delta

## How to read these tables

- Each (model x condition) is its own entrant row. There is deliberately no cross-condition winner picked on validation before the test look: under that scheme a model's two conditions would share one 14-configuration budget while an incumbent spent a whole budget on one row, and the losing condition would still have been test-evaluated to produce the number that was then discarded -- a test look that bought nothing and was never reported. Frozen and full_ft therefore each receive the standard budget, each is test-evaluated exactly once, and both appear in the table.
- 95% intervals are percentile bootstrap over the 27 held-out test **subjects** (the resampling unit is a subject, not a window). The per-entrant column is *marginal and unpaired*; the delta column is *paired*, resampling the same subject indices for both entrants (common random numbers) so the between-entrant correlation is preserved. Ranks and significance are read off the paired column only.
- Each subject's error is averaged over the seeds before resampling, so the bootstrap CI describes subject-to-subject variability at a fixed seed set and **excludes seed variance**. The seed spread is reported beside every mean; a delta smaller than the seed spread is not a result.
- **Do not read a ranking off overlapping per-entrant CIs.** The `95% CI` column is marginal and unpaired; two entrants whose marginal intervals overlap can still differ with a paired interval far from zero, and two whose intervals barely overlap can be a statistical tie. The `paired Δ` column is the one that decides.

## Ranking by horizon

### h = 1

Paired reference: `kahlus_gru` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `kahlus_gru` | incumbent | - | 0.248775 | 0.311022 | [0.225051, 0.407050] | 0.000673 | **reference** | — | — | not_applicable_no_pretraining | False | False |
| 2 | `cbramod__full_ft` | external_pretrained | full_ft | 0.276224 | 0.334928 | [0.246412, 0.433922] | 0.007684 | +0.023906 [+0.018342, +0.029933] | — | no | clean | False | False |
| 3 | `labram_base__full_ft` | external_pretrained | full_ft | 0.282359 | 0.337121 | [0.247558, 0.437429] | 0.000682 | +0.026099 [+0.018100, +0.034807] | — | no | clean | False | False |
| 4 | `gbm` | incumbent | - | 0.271895 | 0.338726 | [0.247293, 0.441467] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 5 | `tuned_ridge_ar` | incumbent | - | 0.289775 | 0.344299 | [0.253027, 0.446616] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 6 | `persistence` | incumbent | - | 0.404677 | 0.494632 | [0.341602, 0.665049] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 7 | `labram_base__frozen` | external_pretrained | frozen | 0.433066 | 0.500393 | [0.405107, 0.603698] | 0.004398 | +0.189372 [+0.154953, +0.233100] | — | no | clean | False | False |
| 8 | `cbramod__frozen` | external_pretrained | frozen | 0.498877 | 0.538054 | [0.438101, 0.649331] | 0.002365 | +0.227032 [+0.201518, +0.255281] | — | no | clean | False | True |
| 9 | `eegpt__full_ft` | external_pretrained | full_ft | 0.504823 | 0.553142 | [0.440251, 0.676313] | 0.013169 | +0.242121 [+0.206001, +0.281301] | — | no | clean | True | False |
| 10 | `eegpt__frozen` | external_pretrained | frozen | 0.597690 | 0.613579 | [0.507118, 0.726738] | 0.004402 | +0.302558 [+0.259726, +0.355376] | — | no | clean | True | True |
| 11 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.080307 | 1.109719 | [0.977053, 1.260371] | 0.002557 | +0.798697 [+0.677834, +0.944861] | — | no | clean | False | True |
| 12 | `biot_prest16__frozen` | external_pretrained | frozen | 1.084940 | 1.121837 | [0.990966, 1.270586] | 0.016706 | +0.810816 [+0.687898, +0.961589] | — | no | clean | False | False |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

### h = 2

Paired reference: `kahlus_gru` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `kahlus_gru` | incumbent | - | 0.317695 | 0.362411 | [0.270577, 0.465584] | 0.001649 | **reference** | — | — | not_applicable_no_pretraining | False | True |
| 2 | `cbramod__full_ft` | external_pretrained | full_ft | 0.337631 | 0.377565 | [0.285533, 0.477716] | 0.004183 | +0.015154 [+0.008618, +0.022566] | 0.001242 | no | clean | False | True |
| 3 | `gbm` | incumbent | - | 0.346980 | 0.385076 | [0.291141, 0.489047] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 4 | `tuned_ridge_ar` | incumbent | - | 0.360410 | 0.396239 | [0.300075, 0.502385] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 5 | `labram_base__full_ft` | external_pretrained | full_ft | 0.354059 | 0.401115 | [0.301879, 0.510179] | 0.004445 | +0.038704 [+0.027247, +0.051189] | 0.0001002 | no | clean | False | False |
| 6 | `persistence` | incumbent | - | 0.452297 | 0.518837 | [0.365099, 0.692151] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 7 | `labram_base__frozen` | external_pretrained | frozen | 0.514754 | 0.560416 | [0.460664, 0.667450] | 0.010274 | +0.198004 [+0.164811, +0.241883] | 4.748e-05 | no | clean | False | False |
| 8 | `eegpt__full_ft` | external_pretrained | full_ft | 0.556613 | 0.589032 | [0.478600, 0.710449] | 0.002832 | +0.226621 [+0.198998, +0.257958] | 4.748e-05 | no | clean | True | True |
| 9 | `cbramod__frozen` | external_pretrained | frozen | 0.569626 | 0.616261 | [0.518149, 0.721502] | 0.003993 | +0.253849 [+0.218749, +0.297730] | 4.748e-05 | no | clean | False | True |
| 10 | `eegpt__frozen` | external_pretrained | frozen | 0.646951 | 0.656469 | [0.548858, 0.774826] | 0.009156 | +0.294057 [+0.253365, +0.347471] | 4.748e-05 | no | clean | True | False |
| 11 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.094921 | 1.124449 | [0.993810, 1.275050] | 0.003388 | +0.762037 [+0.641371, +0.907027] | 4.748e-05 | no | clean | False | True |
| 12 | `biot_prest16__frozen` | external_pretrained | frozen | 1.099249 | 1.132060 | [1.001708, 1.283293] | 0.000688 | +0.769649 [+0.648686, +0.915470] | 4.748e-05 | no | clean | False | True |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

### h = 4

Paired reference: `kahlus_gru` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `kahlus_gru` | incumbent | - | 0.408348 | 0.452718 | [0.363481, 0.553990] | 0.003286 | **reference** | — | — | not_applicable_no_pretraining | False | False |
| 2 | `cbramod__full_ft` | external_pretrained | full_ft | 0.437714 | 0.476316 | [0.383504, 0.583078] | 0.003466 | +0.023598 [+0.015553, +0.032637] | 0.0001489 | no | clean | False | True |
| 3 | `gbm` | incumbent | - | 0.443953 | 0.483825 | [0.391310, 0.589796] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 4 | `labram_base__full_ft` | external_pretrained | full_ft | 0.452087 | 0.490132 | [0.398485, 0.594514] | 0.007756 | +0.037414 [+0.027942, +0.048536] | 4.748e-05 | no | clean | False | False |
| 5 | `tuned_ridge_ar` | incumbent | - | 0.472976 | 0.505238 | [0.414430, 0.608329] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 6 | `labram_base__frozen` | external_pretrained | frozen | 0.601893 | 0.641740 | [0.541293, 0.751105] | 0.008854 | +0.189023 [+0.156583, +0.232979] | 4.748e-05 | no | clean | False | False |
| 7 | `persistence` | incumbent | - | 0.590813 | 0.648675 | [0.496761, 0.825087] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 8 | `eegpt__full_ft` | external_pretrained | full_ft | 0.627688 | 0.661969 | [0.548066, 0.787879] | 0.004457 | +0.209251 [+0.180441, +0.241400] | 4.748e-05 | no | clean | True | False |
| 9 | `cbramod__frozen` | external_pretrained | frozen | 0.647582 | 0.692520 | [0.595774, 0.798328] | 0.011758 | +0.239803 [+0.206038, +0.282244] | 4.748e-05 | no | clean | False | True |
| 10 | `eegpt__frozen` | external_pretrained | frozen | 0.697504 | 0.715101 | [0.600544, 0.839068] | 0.002758 | +0.262383 [+0.225444, +0.311068] | 4.748e-05 | no | clean | True | True |
| 11 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.094044 | 1.125712 | [0.987438, 1.286556] | 0.002263 | +0.672995 [+0.569285, +0.806597] | 4.748e-05 | no | clean | False | True |
| 12 | `biot_prest16__frozen` | external_pretrained | frozen | 1.100518 | 1.135623 | [0.995214, 1.299286] | 0.002231 | +0.682905 [+0.578138, +0.819882] | 4.748e-05 | no | clean | False | True |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

### h = 8

Paired reference: `kahlus_gru` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `kahlus_gru` | incumbent | - | 0.632680 | 0.635661 | [0.541874, 0.735896] | 0.003037 | **reference** | — | — | not_applicable_no_pretraining | False | True |
| 2 | `gbm` | incumbent | - | 0.681481 | 0.673132 | [0.577983, 0.775252] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 3 | `cbramod__full_ft` | external_pretrained | full_ft | 0.686501 | 0.686264 | [0.591443, 0.787928] | 0.009727 | +0.050603 [+0.041847, +0.060129] | 4.748e-05 | no | clean | False | False |
| 4 | `labram_base__full_ft` | external_pretrained | full_ft | 0.692543 | 0.697479 | [0.599215, 0.804881] | 0.005492 | +0.061818 [+0.049837, +0.076384] | 4.748e-05 | no | clean | False | False |
| 5 | `tuned_ridge_ar` | incumbent | - | 0.713797 | 0.725412 | [0.631361, 0.824926] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 6 | `eegpt__full_ft` | external_pretrained | full_ft | 0.781948 | 0.808144 | [0.697173, 0.927665] | 0.005966 | +0.172484 [+0.146982, +0.201388] | 4.748e-05 | no | clean | True | True |
| 7 | `labram_base__frozen` | external_pretrained | frozen | 0.791751 | 0.811037 | [0.703149, 0.930924] | 0.002892 | +0.175376 [+0.143730, +0.215380] | 4.748e-05 | no | clean | False | False |
| 8 | `eegpt__frozen` | external_pretrained | frozen | 0.816583 | 0.821993 | [0.709462, 0.946919] | 0.003444 | +0.186332 [+0.157674, +0.223805] | 4.748e-05 | no | clean | True | True |
| 9 | `cbramod__frozen` | external_pretrained | frozen | 0.824597 | 0.854013 | [0.748081, 0.972366] | 0.004900 | +0.218353 [+0.183910, +0.259199] | 4.748e-05 | no | clean | False | True |
| 10 | `persistence` | incumbent | - | 0.966907 | 0.991174 | [0.833167, 1.153473] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 11 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.090073 | 1.118080 | [0.979826, 1.290044] | 0.001264 | +0.482419 [+0.394865, +0.591500] | 4.748e-05 | no | clean | False | True |
| 12 | `biot_prest16__frozen` | external_pretrained | frozen | 1.093542 | 1.127370 | [0.988473, 1.300701] | 0.004470 | +0.491709 [+0.403528, +0.602273] | 4.748e-05 | no | clean | False | True |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

### h = 16

Paired reference: `kahlus_gru` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `kahlus_gru` | incumbent | - | 0.793454 | 0.825111 | [0.710834, 0.950692] | 0.001500 | **reference** | — | — | not_applicable_no_pretraining | False | True |
| 2 | `gbm` | incumbent | - | 0.834682 | 0.870090 | [0.754759, 0.998452] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 3 | `cbramod__full_ft` | external_pretrained | full_ft | 0.841359 | 0.879679 | [0.762574, 1.010872] | 0.004513 | +0.054567 [+0.045423, +0.063961] | 4.748e-05 | no | clean | False | False |
| 4 | `labram_base__full_ft` | external_pretrained | full_ft | 0.842385 | 0.886702 | [0.768669, 1.018836] | 0.002415 | +0.061591 [+0.047535, +0.076235] | 6.243e-05 | no | clean | False | True |
| 5 | `eegpt__full_ft` | external_pretrained | full_ft | 0.870180 | 0.916932 | [0.797232, 1.049959] | 0.005617 | +0.091821 [+0.074530, +0.110624] | 4.748e-05 | no | clean | True | True |
| 6 | `eegpt__frozen` | external_pretrained | frozen | 0.882368 | 0.943201 | [0.820431, 1.083584] | 0.000823 | +0.118089 [+0.093496, +0.148101] | 4.748e-05 | no | clean | True | False |
| 7 | `tuned_ridge_ar` | incumbent | - | 0.897738 | 0.951832 | [0.836984, 1.082256] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 8 | `labram_base__frozen` | external_pretrained | frozen | 0.932426 | 0.984575 | [0.858585, 1.131256] | 0.002895 | +0.159464 [+0.123007, +0.203554] | 4.748e-05 | no | clean | False | False |
| 9 | `cbramod__frozen` | external_pretrained | frozen | 0.943909 | 1.004950 | [0.883385, 1.146589] | 0.002317 | +0.179839 [+0.144383, +0.221266] | 4.748e-05 | no | clean | False | False |
| 10 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.052465 | 1.133145 | [0.991657, 1.308083] | 0.001884 | +0.308034 [+0.247337, +0.387729] | 4.748e-05 | no | clean | False | True |
| 11 | `biot_prest16__frozen` | external_pretrained | frozen | 1.056023 | 1.134793 | [0.994969, 1.307984] | 0.004305 | +0.309682 [+0.248548, +0.389139] | 4.748e-05 | no | clean | False | True |
| 12 | `persistence` | incumbent | - | 1.543849 | 1.541198 | [1.329219, 1.765043] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

### h = 32

Paired reference: `kahlus_gru` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `cbramod__full_ft` | external_pretrained | full_ft | 0.915750 | 0.970531 | [0.856633, 1.099519] | 0.003832 | -0.001494 [-0.011719, +0.007722] | 0.857 | **tie** | clean | False | False |
| 2 | `kahlus_gru` | incumbent | - | 0.896130 | 0.972025 | [0.853231, 1.107376] | 0.002237 | **reference** | — | — | not_applicable_no_pretraining | False | True |
| 3 | `labram_base__full_ft` | external_pretrained | full_ft | 0.913357 | 0.976017 | [0.858235, 1.110360] | 0.010694 | +0.003992 [-0.005556, +0.012437] | 0.3304 | **tie** | clean | False | True |
| 4 | `gbm` | incumbent | - | 0.923471 | 0.993100 | [0.873990, 1.129225] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | True |
| 5 | `eegpt__full_ft` | external_pretrained | full_ft | 0.933039 | 1.019083 | [0.893501, 1.166865] | 0.002506 | +0.047058 [+0.029513, +0.067933] | 8.397e-05 | no | clean | True | False |
| 6 | `tuned_ridge_ar` | incumbent | - | 0.946351 | 1.021588 | [0.896470, 1.165234] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 7 | `eegpt__frozen` | external_pretrained | frozen | 0.936092 | 1.023321 | [0.894645, 1.175210] | 0.002843 | +0.051296 [+0.028791, +0.078433] | 0.0002542 | no | clean | True | True |
| 8 | `labram_base__frozen` | external_pretrained | frozen | 0.988151 | 1.031810 | [0.908501, 1.174630] | 0.001551 | +0.059785 [+0.041237, +0.078213] | 0.0002532 | no | clean | False | False |
| 9 | `cbramod__frozen` | external_pretrained | frozen | 0.971318 | 1.041945 | [0.916097, 1.188635] | 0.002201 | +0.069920 [+0.048875, +0.093497] | 8.016e-05 | no | clean | False | False |
| 10 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.067528 | 1.139289 | [1.004360, 1.302590] | 0.000559 | +0.167264 [+0.129690, +0.209913] | 4.748e-05 | no | clean | False | False |
| 11 | `biot_prest16__frozen` | external_pretrained | frozen | 1.070803 | 1.144012 | [1.007376, 1.309043] | 0.001841 | +0.171987 [+0.134349, +0.215865] | 4.748e-05 | no | clean | False | True |
| 12 | `persistence` | incumbent | - | 2.127011 | 1.988919 | [1.744251, 2.266364] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

### h = 64

Paired reference: `tuned_ridge_ar` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `labram_base__full_ft` | external_pretrained | full_ft | 0.966050 | 1.015500 | [0.891370, 1.153559] | 0.003585 | -0.000198 [-0.007736, +0.005868] | 0.3304 | **tie** | clean | False | True |
| 2 | `tuned_ridge_ar` | incumbent | - | 0.963951 | 1.015698 | [0.890765, 1.155196] | 0.000000 | **reference** | — | — | not_applicable_no_pretraining | False | False |
| 3 | `kahlus_gru` | incumbent | - | 0.962416 | 1.017738 | [0.891775, 1.158120] | 0.002917 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | True |
| 4 | `cbramod__full_ft` | external_pretrained | full_ft | 0.963099 | 1.018008 | [0.892435, 1.158464] | 0.001247 | +0.002310 [-0.001989, +0.006819] | 0.5278 | **tie** | clean | False | True |
| 5 | `gbm` | incumbent | - | 0.977029 | 1.019900 | [0.891715, 1.162764] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | True |
| 6 | `cbramod__frozen` | external_pretrained | frozen | 0.980274 | 1.033732 | [0.907130, 1.176971] | 0.000179 | +0.018034 [+0.007654, +0.029045] | 0.00191 | no | clean | False | False |
| 7 | `eegpt__frozen` | external_pretrained | frozen | 0.991231 | 1.051478 | [0.923595, 1.195835] | 0.002566 | +0.035780 [+0.022475, +0.050526] | 0.0001715 | no | clean | True | False |
| 8 | `eegpt__full_ft` | external_pretrained | full_ft | 0.993095 | 1.054986 | [0.926279, 1.200759] | 0.001117 | +0.039288 [+0.027706, +0.053948] | 4.748e-05 | no | clean | True | True |
| 9 | `labram_base__frozen` | external_pretrained | frozen | 1.021670 | 1.055384 | [0.925061, 1.207766] | 0.001620 | +0.039686 [+0.015194, +0.067066] | 0.002569 | no | clean | False | True |
| 10 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.088127 | 1.140641 | [0.997396, 1.312052] | 0.000826 | +0.124943 [+0.077270, +0.180998] | 0.0001892 | no | clean | False | False |
| 11 | `biot_prest16__frozen` | external_pretrained | frozen | 1.089903 | 1.141748 | [0.999108, 1.312627] | 0.002448 | +0.126050 [+0.078522, +0.181951] | 0.0002305 | no | clean | False | False |
| 12 | `persistence` | incumbent | - | 2.423631 | 2.321649 | [2.033065, 2.671261] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

### h = 128

Paired reference: `kahlus_gru` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `kahlus_gru` | incumbent | - | 1.051738 | 1.046740 | [0.918422, 1.194330] | 0.001205 | **reference** | — | — | not_applicable_no_pretraining | False | True |
| 2 | `cbramod__full_ft` | external_pretrained | full_ft | 1.058860 | 1.055237 | [0.927281, 1.201511] | 0.004038 | +0.008497 [+0.003391, +0.013716] | 0.02209 | no | clean | False | True |
| 3 | `gbm` | incumbent | - | 1.056842 | 1.057247 | [0.926590, 1.206817] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 4 | `labram_base__full_ft` | external_pretrained | full_ft | 1.064964 | 1.057613 | [0.928017, 1.208836] | 0.003945 | +0.010873 [-0.001063, +0.024531] | 0.3304 | **tie** | clean | False | True |
| 5 | `tuned_ridge_ar` | incumbent | - | 1.064616 | 1.059891 | [0.931960, 1.207028] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |
| 6 | `labram_base__frozen` | external_pretrained | frozen | 1.078832 | 1.076885 | [0.944190, 1.233039] | 0.000360 | +0.030145 [+0.014944, +0.048078] | 0.001552 | no | clean | False | True |
| 7 | `cbramod__frozen` | external_pretrained | frozen | 1.087002 | 1.081334 | [0.949893, 1.234917] | 0.000698 | +0.034594 [+0.022013, +0.049137] | 0.0001715 | no | clean | False | False |
| 8 | `eegpt__full_ft` | external_pretrained | full_ft | 1.104312 | 1.096816 | [0.965515, 1.250242] | 0.002943 | +0.050077 [+0.037427, +0.063801] | 4.748e-05 | no | clean | True | False |
| 9 | `eegpt__frozen` | external_pretrained | frozen | 1.112368 | 1.105380 | [0.969326, 1.265821] | 0.001573 | +0.058640 [+0.040768, +0.080704] | 4.748e-05 | no | clean | True | True |
| 10 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.151617 | 1.145654 | [1.002660, 1.318165] | 0.000443 | +0.098914 [+0.064260, +0.140470] | 0.0001023 | no | clean | False | True |
| 11 | `biot_prest16__frozen` | external_pretrained | frozen | 1.151119 | 1.147064 | [1.007596, 1.314972] | 0.008352 | +0.100324 [+0.067261, +0.137574] | 9.205e-05 | no | clean | False | False |
| 12 | `persistence` | incumbent | - | 2.873092 | 2.835598 | [2.457880, 3.313929] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

### h = 256

Paired reference: `tuned_ridge_ar` (best incumbent at this horizon). Δ = entrant − reference, per test subject; positive Δ means the incumbent has the lower error.

| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |
|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|
| 1 | `tuned_ridge_ar` | incumbent | - | 1.113422 | 1.133158 | [1.005127, 1.286046] | 0.000000 | **reference** | — | — | not_applicable_no_pretraining | False | False |
| 2 | `gbm` | incumbent | - | 1.112080 | 1.133451 | [1.005229, 1.285040] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | True |
| 3 | `kahlus_gru` | incumbent | - | 1.112689 | 1.133693 | [1.004450, 1.287508] | 0.000019 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | True |
| 4 | `labram_base__full_ft` | external_pretrained | full_ft | 1.113474 | 1.135073 | [1.006184, 1.289174] | 0.000424 | +0.001915 [+0.000432, +0.003504] | 0.1261 | no | clean | False | True |
| 5 | `cbramod__full_ft` | external_pretrained | full_ft | 1.113144 | 1.135145 | [1.006509, 1.288833] | 0.000998 | +0.001988 [+0.000740, +0.003421] | 0.02209 | no | clean | False | True |
| 6 | `cbramod__frozen` | external_pretrained | frozen | 1.113871 | 1.135459 | [1.007075, 1.288649] | 0.000030 | +0.002301 [+0.000871, +0.003941] | 0.007389 | no | clean | False | True |
| 7 | `eegpt__full_ft` | external_pretrained | full_ft | 1.114769 | 1.136402 | [1.007625, 1.290097] | 0.000147 | +0.003244 [+0.001610, +0.005203] | 0.0008038 | no | clean | True | True |
| 8 | `eegpt__frozen` | external_pretrained | frozen | 1.114568 | 1.136511 | [1.007588, 1.290448] | 0.000003 | +0.003354 [+0.001590, +0.005438] | 0.001132 | no | clean | True | False |
| 9 | `labram_base__frozen` | external_pretrained | frozen | 1.115056 | 1.136694 | [1.007478, 1.291275] | 0.000786 | +0.003537 [+0.001699, +0.005621] | 0.001552 | no | clean | False | True |
| 10 | `biot_prest16__full_ft` | external_pretrained | full_ft | 1.114255 | 1.136728 | [1.007756, 1.290672] | 0.000751 | +0.003570 [+0.001681, +0.005734] | 0.001582 | no | clean | False | False |
| 11 | `biot_prest16__frozen` | external_pretrained | frozen | 1.115714 | 1.137512 | [1.008721, 1.290757] | 0.000657 | +0.004354 [+0.002488, +0.006605] | 0.0002305 | no | clean | False | False |
| 12 | `persistence` | incumbent | - | 2.156175 | 2.166878 | [1.908886, 2.483339] | 0.000000 | not paired (incumbent) | — | — | not_applicable_no_pretraining | False | False |

Multiplicity: Holm step-down per entrant pair across the claim-eligible horizons; h = 1 is protocol-ineligible and is not in any family. alpha = 0.05.

## Controls

| control | subject | outcome | threshold |
|---|---|---|---|
| `harness_sanity` | AR(2) phi = [0.9, -0.2], 2000 windows, context 127, 500-sample burn-in | PASS | on a stationary AR(2) process a tuned linear model must beat the mean predictor by at least 20%; if it does not, the harness cannot detect linear structure and no ranking it produces is meaningful |
| `harness_sanity` | i.i.d. standard normal, 2000 windows, context 127 | PASS | on a signal with no predictable structure the tuned ridge must not beat the mean predictor by more than 5%; scoring on noise means the harness is broken |
| `harness_sanity` | AR(2) phi = [0.9, -0.2], 2000 windows, context 127, 500-sample burn-in | PASS | on a stationary AR(2) process a tuned linear model must beat the mean predictor by at least 20%; if it does not, the harness cannot detect linear structure and no ranking it produces is meaningful |
| `harness_sanity` | i.i.d. standard normal, 2000 windows, context 127 | PASS | on a signal with no predictable structure the tuned ridge must not beat the mean predictor by more than 5%; scoring on noise means the harness is broken |
| `harness_sanity` | AR(2) phi = [0.9, -0.2], 2000 windows, context 127, 500-sample burn-in | PASS | on a stationary AR(2) process a tuned linear model must beat the mean predictor by at least 20%; if it does not, the harness cannot detect linear structure and no ranking it produces is meaningful |
| `harness_sanity` | i.i.d. standard normal, 2000 windows, context 127 | PASS | on a signal with no predictable structure the tuned ridge must not beat the mean predictor by more than 5%; scoring on noise means the harness is broken |
| `harness_sanity` | AR(2) phi = [0.9, -0.2], 2000 windows, context 127, 500-sample burn-in | PASS | on a stationary AR(2) process a tuned linear model must beat the mean predictor by at least 20%; if it does not, the harness cannot detect linear structure and no ranking it produces is meaningful |
| `harness_sanity` | i.i.d. standard normal, 2000 windows, context 127 | PASS | on a signal with no predictable structure the tuned ridge must not beat the mean predictor by more than 5%; scoring on noise means the harness is broken |
| `index_audit` | @ h1 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h2 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h4 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h8 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h16 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h32 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h64 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h128 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h256 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h1 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h2 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h4 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h8 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h16 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h32 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h64 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h128 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h256 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h1 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h2 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h4 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h8 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h16 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h32 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h64 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h128 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h256 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h1 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h2 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h4 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h8 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h16 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h32 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h64 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h128 | PASS | any window sharing a sample between context and target fails |
| `index_audit` | @ h256 | PASS | any window sharing a sample between context and target fails |
| `label_shuffle` | persistence @ h2 | n/a | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.542756) or the control fails |
| `label_shuffle` | tuned_ridge_ar @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.432492) or the control fails |
| `label_shuffle` | gbm @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.416375) or the control fails |
| `label_shuffle` | kahlus_gru @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.376774) or the control fails |
| `label_shuffle` | eegpt__frozen @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.769) or the control fails |
| `label_shuffle` | labram_base__frozen @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.634947) or the control fails |
| `label_shuffle` | cbramod__frozen @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.678474) or the control fails |
| `label_shuffle` | biot_prest16__frozen @ h2 | **FAIL** | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 1.3231) or the control fails |
| `label_shuffle` | labram_base__full_ft @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.419021) or the control fails |
| `label_shuffle` | biot_prest16__full_ft @ h2 | **FAIL** | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 1.31528) or the control fails |
| `label_shuffle` | eegpt__full_ft @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.668493) or the control fails |
| `label_shuffle` | cbramod__full_ft @ h2 | PASS | training targets permuted across windows; the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.415522) or the control fails |
| `not_implemented` | — | **NOT IMPLEMENTED** | plan section 4 requires a probe of whether each entrant's representation predicts test-subject identity above chance + 0.20 (Identity Trap, arXiv:2606.06647). It is not implemented: the adapters expose a (task, seed, ste |
| `not_implemented` | — | **NOT IMPLEMENTED** | the AGENTS.md control list names a cluster-permutation null. It is not implemented: this task emits one scalar target per window rather than a time-cluster, so the cluster-forming step has no definition here and inventin |
| `random_init_twin` | biot_prest16__frozen @ h1 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__frozen @ h2 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__frozen @ h4 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__frozen @ h8 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__frozen @ h16 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__frozen @ h32 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__frozen @ h64 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__frozen @ h128 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__frozen @ h256 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h1 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h2 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h4 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h8 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h16 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h32 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h64 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h128 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | biot_prest16__full_ft @ h256 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h1 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h2 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h4 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h8 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h16 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h32 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h64 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h128 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__frozen @ h256 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h1 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h2 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h4 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h8 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h16 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h32 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h64 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h128 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | cbramod__full_ft @ h256 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h1 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h2 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h4 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h8 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h16 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h32 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h64 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h128 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__frozen @ h256 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h1 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h2 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h4 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h8 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h16 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h32 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h64 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h128 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | eegpt__full_ft @ h256 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h1 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h2 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h4 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h8 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h16 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h32 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h64 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h128 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__frozen @ h256 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h1 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h2 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h4 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h8 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h16 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h32 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h64 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h128 | **FAIL** | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `random_init_twin` | labram_base__full_ft @ h256 | PASS | a pretrained model that does not beat its own random-init twin shows pretraining bought nothing on this task (plan section 4: 'the most important control in the design'; plan section 5 kill criterion) |
| `time_shift` | persistence @ h2 | n/a | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.542756) or the control fails |
| `time_shift` | tuned_ridge_ar @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.432492) or the control fails |
| `time_shift` | gbm @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.416375) or the control fails |
| `time_shift` | kahlus_gru @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.376774) or the control fails |
| `time_shift` | eegpt__frozen @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.769) or the control fails |
| `time_shift` | labram_base__frozen @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.634947) or the control fails |
| `time_shift` | cbramod__frozen @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.678474) or the control fails |
| `time_shift` | biot_prest16__frozen @ h2 | **FAIL** | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 1.3231) or the control fails |
| `time_shift` | labram_base__full_ft @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.419021) or the control fails |
| `time_shift` | biot_prest16__full_ft @ h2 | **FAIL** | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 1.31528) or the control fails |
| `time_shift` | eegpt__full_ft @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.668493) or the control fails |
| `time_shift` | cbramod__full_ft @ h2 | PASS | training targets rolled 7 windows off their true lag (17.92 s at 100 Hz); the corrupted validation MSE must exceed the honest one by at least 20% (>= 0.415522) or the control fails |

## Tuning budget

- configurations per gradient-trained entrant: 14
- seeds: [0, 1, 2]
- epoch budget: 50
- budget symmetric across gradient entrants (from the ledger): True
- grid subset rule (fixed before the run): 5 (learning rate) x 3 (second axis) = 15 candidate cells, minus the single corner at (largest learning rate, largest second-axis value), giving exactly 14
- attempts logged this run: 7779 (including prior runs into this directory: 7779)
- test-set evaluations this run: 540; cumulative in this directory: 540
- unfreeze-depth axis: swept = False (full_encoder)

## Checkpoints actually loaded

- `biot_prest16` — `/raid/scratch/kgarg/eeg-fm-bench/weights/biot-EEG-PREST-16-channels.ckpt` observed sha256 `40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70` (matches manifest: True); load report: `{"matched_keys": 57, "missing_keys": [], "unexpected_keys": []}`
- `cbramod` — `/raid/scratch/kgarg/eeg-fm-bench/weights/cbramod_pretrained_weights.pth` observed sha256 `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178` (matches manifest: True); load report: `{"matched_keys": 211, "missing_keys": [], "unexpected_keys": []}`
- `eegpt` — `/raid/scratch/kgarg/eeg-fm-bench/weights/eegpt_mcae_58chs_4s_large4E.ckpt` observed sha256 `9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0` (matches manifest: True); load report: `null`
- `labram_base` — `/raid/scratch/kgarg/eeg-fm-bench/weights/labram-base.pth` observed sha256 `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c` (matches manifest: True); load report: `{"checkpoint_entries": 229, "checkpoint_key": "model", "checkpoint_path": "/raid/scratch/kgarg/eeg-fm-bench/weights/labram-base.pth", "dropped_keys": ["lm_head.`

## Shards

Merged from 4 shard run(s). Every table above was recomputed from the union of the shards' per-subject test errors: no shard's own ranking, paired statistics or gate verdict was carried over, and the shards were refused unless they agreed on protocol, split, normalisation, code revision and checkpoint bytes.

| shard dir | run id | commit | entrants carried | horizons | result rows |
|---|---|---|---|---:|---:|
| `/raid/scratch/kgarg/eeg-fm-bench/runs/full-frozen-inc` | `20260821T063435Z-pid3239827-04818917d335` | `04818917d335` | `biot_prest16__frozen`, `biot_prest16__frozen__random_twin`, `cbramod__frozen`, `cbramod__frozen__random_twin`, `eegpt__frozen`, `eegpt__frozen__random_twin`, `gbm`, `kahlus_gru`, `labram_base__frozen`, `labram_base__frozen__random_twin`, `persistence`, `tuned_ridge_ar` | 1, 2, 4, 8, 16, 32, 64, 128, 256 | 108 |
| `/raid/scratch/kgarg/eeg-fm-bench/runs/full-labram-biot-ft` | `20260821T063431Z-pid3239822-04818917d335` | `04818917d335` | `biot_prest16__full_ft`, `biot_prest16__full_ft__random_twin`, `labram_base__full_ft`, `labram_base__full_ft__random_twin` | 1, 2, 4, 8, 16, 32, 64, 128, 256 | 36 |
| `/raid/scratch/kgarg/eeg-fm-bench/runs/full-eegpt-ft` | `20260821T063436Z-pid3239817-04818917d335` | `04818917d335` | `eegpt__full_ft`, `eegpt__full_ft__random_twin` | 1, 2, 4, 8, 16, 32, 64, 128, 256 | 18 |
| `/raid/scratch/kgarg/eeg-fm-bench/runs/full-cbramod-ft` | `20260821T063436Z-pid3239812-04818917d335` | `04818917d335` | `cbramod__full_ft`, `cbramod__full_ft__random_twin` | 1, 2, 4, 8, 16, 32, 64, 128, 256 | 18 |

## Caveats

- **EEGPT model-selection exposure.** Sleep-EDFx was in EEGPT's published downstream suite, so its architecture and hyperparameters were chosen with Sleep-EDF results visible. Its weights are clean of Sleep-EDF, and the `selection exposure` column says True, but the column understates it: this is not a contamination-free row, it is a row whose *design* saw the corpus. Read an EEGPT win as an upper bound.
- **BIOT context handicap.** BIOT PREST-16 consumes a 1.0 s window as a single STFT token, so it sees one spectral snapshot rather than a sequence and has strictly less temporal resolution here than any other entrant. It is also mapped onto **slot 0 of its 16-channel bipolar montage, i.e. the FP1-F7 channel identity**: Sleep-EDF's Fpz-Cz derivation is not in that montage, so the single available channel is presented under another channel's learned identity. Both are structural disadvantages of running it off-task, not evidence about BIOT.
- **EEGPT channel mapping.** Fpz-Cz is presented to EEGPT as **FPZ**, following the upstream Sleep-EDF precedent in `external/vendor/eegpt/downstream`. A bipolar derivation is not the same signal as its anode, and the alternative -- a learned projection from the derivation into EEGPT's channel space -- was **not** used, because a trained projection would add capacity that no other entrant receives.
- **LaBraM `fc_norm` is untrained under the frozen probe.** LaBraM-Base's final normalisation layer is fitted during upstream fine-tuning, so in the frozen condition it runs at its pretrained-initialised statistics. The frozen LaBraM row is therefore a slightly pessimistic reading of the frozen encoder; its full fine-tune row does not carry this caveat.
- **`kahlus_gru` was re-tuned, not copied.** The incumbent GRU here is F8's architecture under *this* benchmark's symmetric budget (same lr axis, epoch budget, patience and validation-selection rule as every FM head), not F8's fixed-gradient-step recipe. Its numbers are therefore **not byte-identical to the published F8 row** and must not be quoted as if they were.
- **Per-model context lengths differ.** Plan section 3.2 gives each FM the largest patch-aligned context not exceeding F8's 1.27 s: eegpt 1.25 s, cbramod 1.00 s, labram_base 1.00 s, biot_prest16 1.00 s, against the incumbents' 1.27 s. Every FM therefore sees *less* history than the incumbents -- a conservative handicap that flatters the incumbents and must be stated beside any incumbent win.
- **Input scaling is a swept hyperparameter for CBraMod and LaBraM.** Both upstreams assume microvolts divided by 100; the harness feeds unitless z-scored windows, so the scale is swept over [0.01, 0.1, 1.0] on validation. It occupies the sweep slot the other two models spend on head width, so the *axes* differ across FM rows even though the configuration count does not.
- **Unfreezing depth was not swept.** 'last N blocks' has no model-agnostic definition across these four upstreams: the adapters wrap heterogeneous encoders (CBraMod criss-cross transformer, EEGPT target-encoder, LaBraM ViT, BIOT linear-attention) whose block containers are named and nested differently, so partial unfreezing needs per-adapter surgery rather than a shared parameter-group rule. Adding it would also double the grid to 28 configurations for the fine-tuned rows only, breaking the equal-configuration-count rule of plan section 3.4 that the whole benchmark rests on. The full-fine-tune rows therefore unfreeze the whole encoder, which is the *most* favourable setting for the external models, so the omission cannot flatter the incumbents.
- **The external models are evaluated off-task.** They are bidirectional masked-infilling encoders and classifiers, not forecasters. A poor number here is evidence about transfer to strictly-future single-sample forecasting, and is not evidence about the model's published capabilities.
- **Declared MDE.** The `powered` gate term uses 0.1 z-scored MSE (`declared_in_driver_pre_data__ratified_by_krish_garg_2026-08-21`). The design document names no MDE; this value was declared in the driver before the data and ratified by Krish Garg on 2026-08-21, before any full-scale test evaluation had been merged or read.

