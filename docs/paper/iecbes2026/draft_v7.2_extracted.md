**Developing an Evidence-Gated Framework for Leakage-Controlled EEG
Forecasting**

Aayushya Patel\*, Krish Garg\*, Sohail Zaidi†

\*IntelliScience Training Institute, †San Jose State University

aayushya@aayushya.net, Krish4740@gmail.com, syed.zaidi@sjsu.edu

*Abstract---* Noninvasive brain recordings are increasingly important
for brain-computer interfaces (BCIs), neurotechnology, and predictive
neuroscience. However, electroencephalography (EEG) signals are
challenging to model because they are noisy, high-dimensional, and
subject-specific. This paper investigates whether past EEG signal
windows can predict future EEG windows under conservative evaluation
criteria. Kahlus-V1, an evidence-gated modeling framework for EEG
future-window forecasting, is presented; it treats observed EEG channels
as partial observations of a latent brain-state representation. The
framework was implemented in Python using PyTorch and includes a latent
neural field, causal field update operator, low-rank pair kernel,
stimulus-conditioning module, EEG/fMRI/behavior observation operators,
uncertainty heads, and a top-level compiler module. Validation
incorporated tensor-shape, modality, configuration, and causal stimulus
checks, along with synthetic tests, ablation studies, held-out splits,
and distributed training. Synthetic falsification experiments verified
recovery of known latent-field processes and failure under signal-free
conditions, while held-out splits reduced data leakage and evaluated
generalization to unseen subjects and datasets. The framework models EEG
channels as observations of a hidden brain-state representation rather
than independent sensor streams. Kahlus-V1 was evaluated on the
Mother-of-All Brain-Computer Interface Benchmarks (MOABB) BNCI 2014_001
motor-imagery EEG dataset and compared with ridge regression. In a
100,000-step batch-matched evaluation, Kahlus-V1 achieved MSE = 3.116,
MAE = 1.297, Pearson *r* = 0.972, and R² = 0.942, compared with ridge
regression (MSE = 7.745, MAE = 2.150, *r* = 0.925, R² = 0.856),
representing approximately a 2.5× reduction in MSE. At face value, these
figures suggested strong future-window forecasting performance. The
implementation is open source and available for research use \[25\].
Post-hoc audit, however, showed that this forecasting task applies a
one-sample shift to a 128-sample window, so the input and the target
share 126 of 127 samples; re-scoring an identical ridge model on the
strictly future sample alone changed its error by a factor of roughly
126. MSE = 3.116 and r = 0.972 are therefore retracted as evidence of
forecasting skill and reported only as a measured artifact of window
overlap. Under an isolated, non-overlapping, subject-held-out
re-evaluation (a single-channel Sleep-EDF horizon sweep and a
BNCI2014_001 single-future-sample check), learned models still beat
persistence, mean-of-trace, and random baselines with a margin that
widens as the horizon grows, but the recurrent model does not outperform
a linear ridge autoregressive baseline. The claimed contribution is the
leakage-controlled evaluation protocol and the evidence gate that
detected this failure in the authors\' own pipeline.

Key Words: Electroencephalography, EEG forecasting, brain-computer
interface, MOABB, BNCI2014_001, ridge regression, leakage control,
Evidence Gates.

I.  INTRODUCTION

Brain signals are valuable because they provide a noninvasive window
into the body's nervous-system activity. In brain-computer interface
(BCI) research, EEG can be used to study how electrical activity changes
when a person intends to make movement, responds to stimuli, and
performs a task \[1-3\]. EEG is also attractive as it is non-invasive,
provides high temporal resolution, and is widely used in open BCI
benchmarks. \[4-6\]. Despite these benefits, EEG signals are difficult
to model because they are noisy, high-dimensional, and different between
subjects. Each scalp channel records a noisy voltage waveform over time,
and useful information is distributed across many channels, including
frontal-central channels such as FC1, FC2, and FC3 and central channels
such as C1, C2, and C3 \[7,8\].

Many EEG machine-learning studies focus on classification, such as
predicting whether a subject intended movements of his/her left hand,
right hand, feet, or tongue movement, supination or pronation \[9\].
Whereas in forecasting, the input is past signal window, and the output
is a future signal window. That makes the paper's core task simple to
state: past EEG window → future EEG window. The quality of the
prediction is measured by a numerical error and correlation against the
true future signal window. Creating paired past/future windows increases
the number of supervised forecasting examples, but it also does prompt
for careful leakage controls as nearby EEG windows can and will be
strongly correlated.

It is important to note the existing EEG models that are traditionally
being used to analyze and develop prediction models for EEG sensor data.
This includes Ridge regression, covariance and Riemannian methods,
convolutional neural networks, recurrent neural networks, transformers,
state-space models, and neural translation models

All these models are being extensively employed to develop predictive
models for EEG sensors under various brain operating conditions with and
without external stimuli to forecast their impacts on the brain
activities. However, there are certain limitations as these techniques
are applied to real data. These limitations are described in the section
below that provide motivation to conduct the research presented in this
paper.

II. RELATED WORK AND MOTIVATION

Existing EEG modeling approaches include linear models, ridge
regression, covariance and Riemannian methods, convolution neural
networks (CNNs), transformers, state-space models (SSMs), neural
operators, and self-supervised EEG representation models \[10-15\].
Table I summarizes the main model families that are relevant to this
work.

Table I. Main Model Families Relevant to the Current Work

  ----------------------------------------------------------------
  **Family**     **Role**                 **Why it matters here**
  -------------- ------------------------ ------------------------
  Linear model   Regularized linear       Strong simple baseline
                 mapping from past EEG    for noisy biomedical
                 signals to future EEG    signals
                 signals                  

  Riemannian /   Uses covariance geometry Traditional EEG BCI
  covariance     for EEG decoding         baseline family

  CNN / TCN      Learns temporal and      Standard deep EEG
                 spatial filters          decoding and sequence
                                          baseline

  RNN / LSTM     Sequence modeling over   Natural fit for EEG
                 time                     windows

  Transformer    Attention over sequences Modern sequence-modeling
                                          baseline inspiration

  SSM / Mamba    Efficient long-sequence  Relevant to the
                 modeling                 forecasting and latent
                                          temporal dynamics
  ----------------------------------------------------------------

These EEG modeling approaches have significantly improved the decoding
and classification processes; however, several limitations remain for
future-state forecasting. Firstly, EEG models can appear stronger than
they truly are when the training and test data splits are not separated
carefully. Due to the nature of EEG recording being often segmented into
adjacent windows, a phenomenon called leakage can occur if windows from
the same recording, subject, or session are split improperly. Secondly,
many EEG models treat individual channels as direct input-output signals
without explicitly motivating a hidden brain state. This can produce
decent prediction results but does not provide a framework to be
extended to other measurement modalities such as fMRI, fNIRS, behavior,
or multimodal biomarkers. Third, complex neural models are sometimes
compared to weak baselines which inflate their success. A fair EEG
forecasting study should include strong simple baselines such as ridge
regression which is already heavily used in biomedical data
applications. Finally, EEG papers can overclaim beyond the evaluated
task, especially when a model performs well on one benchmark but has not
been tested thoroughly across subjects, tasks, or modalities.

Figure 1 presents the overall Kahlus-V1 concept: noninvasive EEG is
treated as an observation of a hidden latent neural field, which is
evolved causally and then decoded into a future EEG window.

This prompted the creation of Kahlus-V1, part inspired by TRIBE v2
\[16\], which predicts fMRI responses across visual, auditory, and
language stimuli using a single tri-modal model. Instead of presenting
Kahlus-V1 as a general brain model, this paper evaluates on a concrete
task: future EEG signal forecasting on the MOABB BNCI2014_001 dataset.
Kahlus v1 treats all observed EEG signal channels as individual
measurements of an overall brain-state representation, then uses that
representation to predict the future EEG signal waveforms. This framing
is inspired by dynamic systems, state-space models, neural operators,
and observation models. However, current evidence is intentionally
restricted to the EEG future-state forecasting. The fundamental purpose
of the evidence-gated design is to ensure this paper only reports the
data that can be supported and blocks broader claims when other tasks
fail.

For this paper, the key baseline is ridge regression. Ridge regression
\[17\] is a linear model that adds a penalty to prevent overly large
weights. It is simple, regularized, fast, and often strong on noisy
biomedical data including EEG signals.

A common challenge in brain-signal machine learning is the risk of
overclaiming performance due to data leakage. A model may appear highly
accurate if the data split is not sufficiently strict, if data from the
same subject appears in both the training and test sets, or if
comparisons are made against weak baselines \[18-19\]. The proposed
framework is specifically designed to address this limitation by
enforcing rigorous evaluation protocols and minimizing opportunities for
leakage.

Therefore, Kahlus-V1 includes evidence gates: the system must report
baselines, metrics, leakage checks, and task failures before the paper
decides what claim is allowed. The major question asked is: Can a model
predict the future brain-signal activity using strict leakage-controlled
evaluation and evidence gate that prevent unsupported claims?

A.  RESEARCH OBJECTIVES

The main objective of the work presented in this paper is to evaluate
the effectiveness and whether Kahlus-V1 can forecast future EEG signal
windows from past EEG signal windows on the MOABB BNCI2014_001
motor-imagery dataset under leakage-aware evaluation and conservative
claim gates.

For this purpose, an evidence-gated forecasting framework was developed
and called Kahlus-V1. The key novelty of this compiler is that it does
not depend on any existing off-the-shelf models. The basic idea behind
developing Kahlus-V1 was to treat EEG, fMRI, fNIRS, behavior, and other
brain data measurements as different observations of a hidden brain
state. The current paper validates the first practical part of this
framework: future-state forecasting on EEG data. It is inspired by prior
work in neural operators \[12,20,21\], state-space models \[22 - 23\],
and observation models \[21\], but the current implementation is
evaluated only on EEG future-state forecasting. To perform the target
objectives, EEG data was obtained from the Mother of All Brain Computer
Interface Benchmark (MOABB) \[3\].

The proposed system compares Kahlus-V1 (novel) against a ridge baseline
on future EEG forecasting. Kahlus-V1 also combines training with split
manifests, leakage audits, identity-risk diagnostics, baseline tables,
metric summaries, model cards, and evidence gates.

The objectives of the proposed work are summarized as:

a.  To provide a clear future EEG signal forecasting task definition for
    MOABB BNCI2014_001 dataset.

b.  To provide a proposed framing that treats EEG as an observation of a
    hidden brain-state field;

c.  To provide a comparison against ridge regression on the same
    forecasting task;

d.  To develop evidence gates that keep the paper from claiming more
    than the data support.

![](media/image1.png){width="3.7758103674540684in" height="6.4in"}

**Figure 1. Kahlus-V1 concept. (A) Noninvasive scalp EEG provides the
input window. (B) The model represents brain state as a latent field
that is inferred rather than measured. (C) Forecast error against
baselines across prediction horizons on held-out subjects.**

III\. DATA SOURCES AND PRELIMINARY ANALYSIS

The present study uses the MOABB implementation of the BNCI2014_001
motor-imagery EEG dataset \[3\]. This public dataset contains EEG
recordings from 9 healthy subjects performing a cue-based BCI paradigm
involving four motor imagery tasks: imagination of left-hand,
right-hand, both-feet, and tongue movement. Two sessions were recorded
per subject on separate days, each comprising 6 runs of 48 trials (12
trials per class), yielding 288 trials per session. EEG signals were
acquired using 22 Ag/AgCl electrodes sampled at 250 Hz,
bandpass-filtered between 0.5 Hz and 100 Hz with an additional 50 Hz
notch filter to suppress power-line noise. Each trial followed a fixed
cue paradigm: a fixation cross appeared at t=0 s, a directional arrow
cue was presented at t=2 s for 1.25 s, and subjects performed the
imagery task until t=6 s with no real-time feedback provided.

Raw EEG recordings were segmented into non-overlapping windows of 128
samples (stride = 128). At 250 Hz this corresponds to approximately
0.512 s per window. The forecasting task uses a one-sample shift: input
and target, so each input and target contains 127 samples. This
formulation is heavily autocorrelation-exposed by design; the evidence
gate accounts for this by requiring comparison against both ridge
regression and a persistence baseline before any claim is supported.

A subject-held-out split was used: subjects 1, 3, 4, 5, 7, 9 for
training (12,096 windows); subject 2 for validation (2,016 windows);
subjects 6 and 8 for testing (4,032 windows). The prepared evidence
bundles report 2,592 events and 18,144 total windows across all splits.
Because this is a public research dataset, participants are referred to
as subjects, not patients. The paper does not study clinical diagnosis,
treatment response, or patient monitoring.

Kahlus-V1 visual abstract. The framework infers a latent field from
observed signals, renders task-relevant measurements, and restricts
claims through an evidence gate. In this paper, only the EEG
future-window forecasting path is evaluated.

![](media/fig2_architecture.png){width="4.405112642169729in"
height="6.4in"}

*Figure 2: Kahlus-V1 architecture. Input EEG (22 channels, 128 samples)
is encoded into a latent neural field via a 12-layer GRU (384 dims),
evolved causally, and decoded into a future EEG prediction. Only the EEG
forecasting path is evaluated in this paper.*

IV. PROPOSED METHOD: THE KAHLUS-V1 FRAMEWORK

Kahlus v1 represents brain recordings as observations of a hidden
brain-state process. Put simply, scalp EEG sensors show only part of the
underlying neural activity. Kahlus-V1 learns an internal representation
from the observed EEG signal and uses that representation to forecast
the next EEG window.

The architecture of this evaluated path is shown in Figure 2, and the
component-level configuration is listed in Table II.

The evaluated path in this paper is

![](media/image7.png){width="3.4583333333333335in"
height="0.9548611111111112in"}

The broader multimodal-observation view can be written as \[12\]

*Ym = Om(Fs, As, U) + εm* (1)

where Fs is a subject-conditioned hidden neural field, As represents
subject or anatomical information when available, U represents task or
stimulus context, ε_m represents measurement noise, and Om is an
observation operator that maps the hidden field to a measured signal. In
this paper, Ym is EEG. Other measurement types such as fMRI, fNIRS, and
behavior are future directions and are not claimed as current evidence.

Table II details the components of the evaluated Kahlus v1 instance. The
field inference module is implemented as a 12-layer GRU with input and
hidden size 384, batch-first layout, and no dropout. The observation
operator for EEG forecasting is a single linear readout layer mapping
the 384-dimensional latent field to the 22 EEG sensor channels.

Kahlus-V1 Architecture Components: Kahlus-V1 dataflow. Inputs are mapped
through field inference and field dynamics before being rendered into
measured outputs. The current validated path in this paper is observed
EEG latent field EEG future-window forecast; other rendered outputs
remain future directions.

Table II. Model (Kahlus V1) Configuration

  ---------------------------------------------------------------------------
  **Component**    **Type**            **Input**    **Output**     **Params**
  ---------------- ----------------- ------------- ------------- ------------
  qϕ EEG encoder   Linear + Conv1d    \[B,T,22\]    \[B,T,384\]       600,192
                   residual                                      

  Modality         Learned vector         ---       \[1,1,384\]           384
  embedding                                                      

  Φθ backbone      12-layer GRU       \[B,T,384\]   \[B,T,384\]    10,644,480

  Oα forecast head Linear             \[B,T,384\]   \[B,T,22\]          8,470

  Reconstruction   Linear             \[B,T,384\]   \[B,T,22\]          8,470
  head                                                           

  Readout head     Linear             \[B,T,384\]   \[B,T,22\]          8,470

  Projection head  MLP 384 → 384 →     \[B,384\]     \[B,32\]         160,160
                   32                                            

  Subject adapter  MLP 384 → 48 →     \[B,T,384\]   \[B,T,384\]        37,296
                   384                                           

  Total trainable                                   11,467,922   
  ---------------------------------------------------------------------------

V.  EXPERIMENTATION PROTOCOL

The experiment compares Kahlus-V1 with ridge regression and a
persistence baseline on the same future EEG forecasting task. The
persistence baseline predicts the future window as identical to the past
input window; it requires no training and serves as a natural
autocorrelation control.

Kahlus-V1 was trained using AdamW with learning rate 2e-3 and batch size
128 (effective batch 1024 with gradient accumulation steps of 8). No
learning rate schedule was applied. Training used seed 0 and ran for
100,000 steps on a 3-GPU NVIDIA A100-SXM4-80GB setup before failing
during DDP/NCCL finalization; metrics were recovered from the 100k-step
checkpoint. Ridge regression used with no cross-validation tuning. The
recovered run did not produce per-window prediction arrays, so
per-subject breakdowns and paired significance tests are not reported
from this artifact; these are left to future cleaner training runs.

The evaluation reports mean squared error (MSE), mean absolute error
(MAE), Pearson correlation coefficient, and coefficient of
determination. Lower MSE and MAE indicate lower numerical error. Higher
Pearson indicates stronger correlation between predicted and true future
signals. Higher indicates greater explained variance. Negative indicates
a failed prediction relative to a simple mean-like baseline.

Figure 3 summarizes the evidence-gate logic and the leakage controls
applied to this evaluation.

Recovered batch-matched evidence status. The evidence gate permits the
narrow future-forecasting claim while blocking broad model-superiority
and masked-reconstruction claims.

![](media/image4.png){width="4.894234470691163in" height="6.4in"}

Figure 3: Evidence gates and leakage controls. Subject-held-out splits
prevent leakage, and the input and target windows share no samples. On
held-out Sleep-EDF subjects (single channel, 100 Hz) at a 16-sample
horizon, the recurrent model reaches MSE 0.690 against ridge regression
at 0.822, so the two remain close. The evidence gate approves only the
narrow baseline-improvement claim.

EVIDENCE GATE AS A GUARDRAIL The evidence gate is not presented as the
primary model contribution. It is a scientific guardrail. It does not
improve the model's predictive performance. Instead, it restricts what
the paper is allowed to conclude. In this study, the gate allows only
the narrow isolated-evaluation baseline-improvement claim reported in
Section VII, while blocking claims about clinical diagnosis, emotion
prediction, fMRI prediction, fNIRS prediction, general brain
foundation-model status, or broad model superiority.

VI. RESULTS

Recovered 100,000-step batch-matched task metrics are reported below.
Masked reconstruction fails and blocks broader claims, and Section VII
shows the forecasting comparison itself is overlap-dominated and is
retracted.

Kahlus-V1 achieved MSE = 3.116, compared with ridge regression at MSE =
7.745 and the persistence baseline at MSE = 8.204. The persistence
baseline---which predicts the future window as the past window
unchanged---achieved Pearson = 0.924, nearly identical to ridge
regression (= 0.925), confirming that EEG windows are strongly
autocorrelated. Kahlus-V1's higher correlation (= 0.972) and
substantially lower MSE initially suggested predictive structure beyond
autocorrelation alone; Section VII shows this interpretation does not
survive the overlap audit. MSE and MAE measure how numerically close the
predicted future EEG window is to the true future EEG window. Lower
values mean smaller waveform error. Pearson measures whether the
predicted and true signals rise and fall together over time. measures
how much variance in the true future signal is explained by the
prediction. In this recovered setup, Kahlus-V1 produced lower numerical
errors and higher correlation than ridge regression on the overlapping
formulation. Section VII shows that this apparent advantage is an
artifact of window overlap; it is not interpreted as forecasting skill,
nor as evidence of clinical diagnosis or general brain understanding.

Table III reports the full metric comparison between Kahlus-V1, ridge
regression, and the persistence baseline on the overlapping formulation
described in Section III, together with the isolated
single-future-sample re-evaluation described below.

Table III. Kahlus-V1 results comparison with the baseline (Ridge Model)

  --------------------------------------------------------------------------------------------
  **Task**         **Model**      **MSE**   **MAE**   **Pearson   **R²**  **Interpretation**
                                                         r**              
  ---------------- ------------- --------- --------- ----------- -------- --------------------
  Future EEG       Kahlus v1       3.116     1.297      0.972     0.942   Retracted:
  forecasting                                                             overlap-dominated
                                                                          (Section VII)

  Future EEG       Ridge           7.745     2.150      0.925     0.856   Strong baseline
  forecasting      regression                                             (overlapping
                                                                          formulation)

  Future EEG       Persistence     8.204      ---       0.924     0.847   Autocorrelation
  forecasting      baseline                                               control

  Masked           Kahlus v1      53.977     5.691     −0.013     −0.006  Failed task; broad
  reconstruction                                                          claim blocked

  Masked           Ridge           7.787     2.002      0.952     0.855   Ridge dominates
  reconstruction   regression                                             masked
                                                                          reconstruction
  --------------------------------------------------------------------------------------------

## VII. Supported and Unsupported Claims

Table IV. Claims supported by the recovered evidence.

  ------------------------------------------------------------------
  **Claim**            **Status**   **Evidence in this paper**
  -------------------- ------------ --------------------------------
  The protocol,        Supported    They detected and blocked an
  leakage controls,                 overlap-inflated result produced
  and evidence gate                 by this same pipeline (Section
  work as designed                  VI)

  Learned models       Supported    Isolated non-overlapping
  recover                           evaluation on held-out Sleep-EDF
  short-horizon                     subjects: MSE 0.690 at horizon
  predictive structure              16 (160 ms) against 1.334 for
  in EEG                            persistence and 1.264 for
                                    mean-of-trace

  The latent-state     Supported    The evaluated path in Fig. 2
  framing is                        runs as an executable pipeline
  implemented end to                
  end                               

  The                  Retracted    Input and target shared 126 of
  overlapping-window                127 samples; an identical ridge
  result is evidence                scored MSE 0.0018 overlapping
  of forecasting skill              and 0.227 on the strictly future
                                    sample

  Kahlus-V1            Not          At horizon 1 (BNCI2014_001,
  outperforms ridge    supported    single channel, two test
  regression on future              subjects), ridge 0.517 against
  EEG forecasting                   0.568; the bootstrap interval on
                                    the difference, \[−0.086,
                                    −0.016\], favours the linear
                                    baseline

  Kahlus-V1            Not          MSE 53.977 with Pearson r below
  generalizes to       supported    zero (Table III)
  masked                            
  reconstruction                    

  Clinical,            Not claimed  No diagnosis, monitoring,
  fMRI/fNIRS, or                    seizure, or emotion decoding was
  foundation-model                  evaluated
  capability                        

  The result is        Not          The recovered run stored no
  statistically        supported    per-window predictions, the
  powered                           isolated re-evaluation used only
                                    two held-out subjects, and a
                                    nine-subject BNCI sweep cannot
                                    distinguish the models at any
                                    horizon
  ------------------------------------------------------------------

Table IV summarizes the claims supported by the recovered evidence. This
table is included to address the main interpretability concern: the
numerical result has meaning only relative to the evaluated task.

Claim interpretation from the current evidence.

The central numerical result of this paper must be read against the task
definition given in Section III, not in isolation. The forecasting task
used for the recovered 100,000-step run applies a one-sample shift to a
128-sample window, so the input spans samples 0-126 and the target spans
samples 1-127. The input and the target therefore share 126 of their 127
samples. Under this formulation a model can reach a very low error by
learning the index shift, that is, by mapping input sample k+1 onto
output sample k, without forecasting anything that it has not already
observed.

This was tested directly rather than assumed. Holding the model, the
data, and the training procedure fixed and changing only the scoring
window, on subject-held-out Sleep-EDF data a ridge regression scored MSE
0.0018 on the overlapping full-window target and MSE 0.227 when scored
only on the strictly future sample. That is a factor of roughly 126 from
an identical model, which matches the 126 of 127 shared samples almost
exactly. MSE = 3.116 and Pearson r = 0.972 are therefore treated as a
measurement of window overlap rather than as evidence of forecasting
skill, and are not advanced as a performance claim for Kahlus-V1.

A forecasting task is only a forecast when the target window shares no
samples with the input window. For an input of width W and a horizon H,
the two windows overlap by W - H samples whenever H is smaller than W.
The evaluation reported here consequently scores only the strictly
future sample, so that input and target are disjoint by construction.
The induced inflation is bounded: a window-scored error metric can
overstate strictly-future error by at most a factor of W/H, a bound
verified on 286 window-horizon configurations across eleven signals with
zero violations; at the W = 128, H = 1 configuration used here the bound
predicts a factor of 127, matching the measured factor of roughly 126.

Under that isolated evaluation, on subject-held-out single-channel
Sleep-EDF recordings (100 Hz), both learned models clearly beat the
trivial baselines, and the margin over persistence widens as the
forecast horizon grows. At a horizon of 16 samples (160 ms) the
recurrent model reaches MSE 0.690 against 1.334 for persistence, an
improvement of nearly a factor of two. This indicates predictive
structure beyond simply repeating the most recent value.

The honest limitation is that the recurrent model only matches a linear
ridge autoregressive baseline rather than clearly beating it. On a
subject-held-out single-channel evaluation of BNCI2014_001 at a horizon
of one sample, ridge reached MSE 0.517 and the recurrent model reached
0.568, and a subject-cluster bootstrap confidence interval on the
difference was \[-0.086, -0.016\], which excludes zero in favour of the
linear baseline. The nonlinear architecture is therefore not claimed to
outperform a linear model on this task. A subsequent nine-subject
leave-one-subject-out sweep on BNCI2014_001 at matched training budget,
with the ridge autoregressive baseline tuned per horizon, reached the
same verdict with more power: across nine horizons (10-2560 ms), none of
the 81 fold-horizon comparisons was significant (0 of 9 horizons at p \<
0.05, every per-horizon confidence interval including zero, largest mean
gap 0.0181 MSE), so this corpus cannot distinguish the two models at any
horizon.

Accordingly, the supported claims of this paper are the following.
First, the evaluation protocol, the leakage controls, and the evidence
gate are working as designed, because they detected and blocked an
overlap-inflated result generated by this same pipeline. Second, under
an isolated non-overlapping evaluation on held-out subjects, learned
models recover genuine short-horizon predictive structure in EEG beyond
persistence, mean-of-trace, and random baselines. Third, the
latent-state framing is implemented and executable end to end.

The unsupported claims are equally explicit. This paper does not claim
that Kahlus-V1 outperforms ridge regression on future EEG forecasting.
It does not claim clinical diagnosis, treatment response, patient
monitoring, seizure prediction, or emotion decoding. It does not claim
fMRI or fNIRS prediction, general brain foundation-model status, or
broad model superiority. It does not claim a statistically powered
result, because the recovered run produced no per-window prediction
arrays and the isolated re-evaluation used only two held-out test
subjects, which is below the number of clusters required for a stable
confidence interval; the nine-subject sweep is powered enough only to
report indistinguishability, not superiority in either direction.

> VIII\. Conclusions and Discussion

This paper set out to test one concrete question: whether past EEG
windows can predict future EEG windows under leakage-controlled
evaluation and conservative claim gates. The most informative outcome of
the work was not the magnitude of the headline metric but the fact that
the evaluation machinery rejected it.

The recovered 100,000-step run reported MSE = 3.116 and Pearson r =
0.972 against a ridge baseline at MSE = 7.745, which appears to be a
substantial improvement. An audit of the task definition showed that the
input and target windows shared 126 of 127 samples, so a model could
achieve most of that score by learning a one-sample index shift.
Re-scoring an identical ridge model on the strictly future sample
changed its error by a factor of roughly 126. The apparent result was
therefore an artifact of the task construction, and it is reported as
such rather than as a performance claim.

Re-running the comparison on a properly isolated task changes the
conclusion but does not empty it. On held-out Sleep-EDF subjects,
learned models still beat persistence, mean-of-trace, and random
baselines at every horizon tested, and the advantage over persistence
grows with the forecast lead. Short-horizon EEG forecasting therefore
does contain recoverable structure. What the evidence does not yet
support is that the nonlinear recurrent architecture earns its
additional complexity over a linear autoregressive model, since the two
are statistically indistinguishable or the linear model is slightly
ahead.

Three limitations bound these conclusions. The distributed training run
terminated during DDP/NCCL finalization, so metrics were recovered from
a checkpoint and no per-window prediction arrays survived; paired
significance tests and per-subject breakdowns could not be computed from
that artifact. The isolated re-evaluation used two held-out test
subjects, which is insufficient for a powered claim; a later
nine-subject leave-one-subject-out sweep on BNCI2014_001 confirmed only
that the models cannot be distinguished at any horizon. The isolated
BNCI2014_001 comparison used a single channel, whereas the full model
consumes 22 channels, so a multi-channel replication remains
outstanding.

The wider point is methodological. An overlapping input and target
window is easy to introduce, produces correlations near 0.97 that look
like success, and is difficult to detect from summary metrics alone. The
defence is structural rather than statistical: assert that the input and
target index sets are disjoint, and compare against a persistence
baseline that exposes autocorrelation directly. The demonstration that
an evidence gate caught a false positive produced by the authors\' own
pipeline is regarded as the more durable contribution of this work, and
the invalidated number is reported openly so that the failure mode is
reusable by others.

IX\. Future Work

The long-term promise of better brain-signal models, including future
work of Kahlus, is larger than one benchmark. This includes more
reliable forecasting that could support improved neurotechnology,
assistive interfaces, experimental neuroscience and safer biomedical
signal interpretation. The present paper does NOT make those clinical
claims and retains those for future work under Kahlus-V2 and V3. This
paper tests a first, concrete question: can Kahlus-V1 predict future EEG
signal data from past EEG signal data. Future work will test whether the
same evidence-gated framework transfers to stricter subject-held-out
settings, additional EEG tasks, and later multimodal benchmarks.

REFERENCES

\[1\] G. Pfurtscheller and C. Neuper, \"Motor imagery and direct
brain-computer communication,\" Proceedings of the IEEE, 2001.

\[2\]. Nurse ES, Karoly PJ, Grayden DB, Freestone DR. A Generalizable
Brain-Computer Interface (BCI) Using Machine Learning for Feature
Discovery. PLoS One. 2015 Jun 26;10(6):e0131328. doi:
10.1371/journal.pone.0131328. PMID: 26114954; PMCID: PMC4482677.

\[3\] F. Lotte et al., \"A review of classification algorithms for
EEG-based brain-computer interfaces,\" Journal of Neural Engineering,
2007.

\[4\] V. Jayaram and A. Barachant, \"MOABB: trustworthy algorithm
benchmarking for BCIs,\" Journal of Neural Engineering, vol. 15, no. 6,
p. 066011, 2018.

\[5\] MOABB Consortium, \"The largest EEG-based BCI reproducibility
study for open science: the MOABB benchmark,\" arXiv:2404.15319, 2024.

\[6\] C. Brunner et al., \"BNCI Horizon 2020: towards a roadmap for the
BCI community,\" Brain-Computer Interfaces, 2015.

\[7\] A. Craik, Y. He, and J. L. Contreras-Vidal, \"Deep learning for
electroencephalogram classification tasks: a review,\" Journal of Neural
Engineering, 2019.

\[8\] Y. Roy et al., \"Deep learning-based electroencephalography
analysis: a systematic review,\" Journal of Neural Engineering, 2019.

\[9\] V. J. Lawhern et al., \"EEGNet: a compact convolutional neural
network for EEG-based brain-computer interfaces,\" Journal of Neural
Engineering, 2018.

\[10\] R. T. Schirrmeister et al., \"Deep learning with convolutional
neural networks for EEG decoding and visualization,\" Human Brain
Mapping, 2017.

\[11\] F. Pedregosa et al., \"Scikit-learn: machine learning in
Python,\" Journal of Machine Learning Research, 2011.

\[12\] L. Lu et al., \"Learning nonlinear operators via DeepONet based
on the universal approximation theorem of operators,\" Nature Machine
Intelligence, 2021.

\[13\] D. Kostas, S. Aroca-Ouellette, and F. Rudzicz, \"BENDR: using
transformers and a contrastive self-supervised learning task to learn
from massive amounts of EEG data,\" Frontiers in Human Neuroscience,
2021.

\[14\] H. Banville et al., \"Uncovering the structure of clinical EEG
signals with self-supervised learning,\" Journal of Neural Engineering,
2021.

\[15\] A. Paszke et al., \"PyTorch: an imperative style,
high-performance deep learning library,\" Advances in Neural Information
Processing Systems, 2019.

\[16\] S. d\'Ascoli, J. Rapin, Y. Benchetrit, T. Brooks, K. Begany, J.
Raugel, H. Banville, and J.-R. King, \"A foundation model of vision,
audition, and language for in-silico neuroscience,\" arXiv:2605.04326,
2026.

\[17\] A. E. Hoerl and R. W. Kennard, \"Ridge regression: biased
estimation for nonorthogonal problems,\" Technometrics, 1970.

\[18\] G. Brookshire et al., \"Data leakage in deep learning studies of
translational EEG,\" Frontiers in Neuroscience, vol. 18, 1373515, 2024.

\[19\] F. Del Pup et al., \"The role of data partitioning on the
performance of EEG-based deep learning models in supervised
cross-subject analysis,\" Computers in Biology and Medicine, 2025.

\[20\] Z. Li et al., \"Fourier neural operator for parametric partial
differential equations,\" International Conference on Learning
Representations, 2021.

\[21\] K. J. Friston, L. Harrison, and W. Penny, \"Dynamic causal
modelling,\" NeuroImage, 2003.

\[22\] A. Gu, K. Goel, and C. Re, \"Efficiently modeling long sequences
with structured state spaces,\" International Conference on Learning
Representations, 2022.

\[23\] A. Gu and T. Dao, \"Mamba: linear-time sequence modeling with
selective state spaces,\" arXiv:2312.00752, 2023.

\[24\] M. Tangermann et al., \"Review of the BCI Competition IV,\"
Frontiers in Neuroscience, vol. 6, p. 55, 2012.

\[25\] A. Patel, K. Garg, and S. Zaidi, \"Kahlus-V1: evidence-gated EEG
forecasting framework (source code),\" GitHub repository, 2026.
\[Online\]. Available: https://github.com/ptlnextdoor/Kahlus-V1
