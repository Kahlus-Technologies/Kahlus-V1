# Meeting: Prof. Syed Zaidi paper review (IECBES 2026)

**Date:** 2026-07-27, 6:49 PM - 7:29 PM PT
**Duration:** 39 min 54 s
**Participants:** Prof. Syed Zaidi (screen share, most of the talking), Aayushya, Krish Garg
**Artifact under review:** `Paper - Aayushya - Krish - Zaidi - July 27-2026 version 5.docx` (Word, ~3,224-3,258 words, 10-11 pages)
**Target venue:** IECBES 2026, Malaysia (confirmed on screen: Drive folder `IEEE- IECBES2026- Mal... > Aayushya and Krish`)
**Source recording:** `~/Desktop/Screen Recording 2026-07-27 at 6.49.05 PM.mov` (3.2 GB, 2944x1930 @ 60fps)

---

## Overall

A working session, not a critique session. Zaidi drove the whole meeting from his own screen with the
Word doc open, going top-to-bottom through the paper and narrating fixes as he made them. He was
consistently positive on the substance ("this is a great paper", "I really believe it will be accepted")
and spent essentially all forty minutes on **presentation**: reference verification, figure legibility,
and cross-referencing.

The single most important thing about this meeting is what did **not** come up.

> **Zaidi never questioned the headline result.** MSE 3.116 / Pearson r = 0.972 sat on screen in the
> Figure 2 caption at [18:00] and passed without comment. The overlap-dominated forecasting task that
> produced it -- the one this repo has already retired in `docs/CLAIMS.md` and
> `docs/results/a100-run-history.md` as `future_forecasting_INVALID` -- was never discussed.

So the review he gave is a **formatting and citation review**. It is not a validity review, and it
should not be mistaken for one. The scientific problem in the paper is still unaddressed and Zaidi
does not currently know it exists. Raising it is on us.

## Action items

| # | Item | Source | Status |
|---|------|--------|--------|
| 1 | Collect every cited PDF into one shared reference folder | [00:00] | open |
| 2 | Verify refs [7], [8], [9], [10-15], [16], [17], [18], [19], [21], [22], [23] are real and topically relevant | [00:50]-[08:00] | open |
| 3 | Hyperlink references in the bibliography | [02:36] | agreed, open |
| 4 | Rebuild Figure 1 as grouped Word shapes that fit a half column and stay readable | [09:02]-[15:22] | demoed by Zaidi |
| 5 | Rebuild Figure 3 the same way (2x2 grouped, larger fonts, arrows + labels) | [16:00]-[17:53] | demoed by Zaidi |
| 6 | Call Figure 1, Figure 2, Figure 3 in the main text -- none are currently cited | [17:55]-[20:31] | open |
| 7 | Call Table 3 in main text; renumber the mislabelled "Table 4" to Table 3 | [21:02]-[22:10] | open |
| 8 | Make Table 3 a picture, but a *readable* one | [22:10] | open |
| 9 | Write the "supported vs unsupported claims" / claim-evidence section | [22:32] | open |
| 10 | Write Conclusions and Discussion (currently an empty heading) | [23:06] | open |
| 11 | Decide where the sweep figures go: move into Results + discuss them, or cut | [23:13]-[25:33] | open |
| 12 | Keep filename `version 5 July 27`; do not rename | [26:00], [28:41] | instruction |
| 13 | Write new additions in **red font** so Zaidi can see the diff | [28:16] | instruction |
| 14 | Single column while drafting; do not restructure sections | [27:54], [31:00] | instruction |
| 15 | Submit SCUR abstract (deadline **30 July 2026**, ~250 words) | [33:41]-[34:01] | urgent |
| 16 | Submit IFORE abstract (August, 1500 characters) | [33:41]-[34:01] | open |

**Not on Zaidi's list, but higher priority than all of it:** decide what to do about the invalid
headline number before this goes to reviewers.

---

## Summary by section

### Reference hygiene (00:00 - 08:15)

This was the longest single theme. Zaidi's process, stated explicitly: for every citation, open it in
Google, confirm it exists, confirm it is actually about what the sentence claims, and save the PDF into
a shared folder.

> "I don't want irrelevant references. They will immediately be destroyed." [00:39]

His reasoning is defensive: if a reviewer challenges a citation, you want the PDF already in hand rather
than re-doing the search months later.

He went through them live on screen. Visible in the recording at [02:00] is a Google search for
`A. Craik, Y. He, and J. L. Contreras-Vidal, "Deep learning for electroencephalogram classification
tasks: a review"` (J. Neural Eng. 2019) -- that is reference [7], and it checked out. He questioned the
grouping of [7]-[9] and [10]-[15], wanting [10]-[15] to genuinely be the CNN / SSM / self-supervised
model papers the sentence claims they are. He verified [16] as TRIBE ("1000 hours of MRI across seven
subjects" -- likely garbled ASR for fMRI/MEG hours) and confirmed [17] as the ridge regression citation.

Aayushya offered to email all the PDFs immediately. Krish or Aayushya proposed hyperlinking the
references; Zaidi liked it ("that will be really great").

### Table I and the related-work framing (04:08 - 05:10)

Zaidi was happy here. He specifically praised Table I (model families and their relevance to this work):
"This is good. I like that table." The Related Work and Motivation section reads correctly to him -- the
limitations enumerated there justify the paper's existence, which is the job that section has to do.

### Research objectives (07:15 - 09:00)

He questioned why references [1]-[3] appear in the objectives section and suspected they belong
elsewhere. He also flagged a general principle:

> "The more you put them, the more difficult you make." [08:15]

He then had Aayushya and Krish read the four objectives aloud to confirm they were right:

1. Define a clear future-EEG-signal forecasting task for the MOABB dataset
2. Propose the Neural Field Compiler framing (EEG as observation of a hidden brain-state field)
3. Compare against ridge regression on the same forecasting task
4. Develop evidence gates that keep the paper from claiming more than the data supports

Objective 4 is worth flagging: the paper explicitly promises evidence gates that prevent overclaiming,
while the headline number in Figure 2's caption is one this repo has already ruled invalid. That is an
internal contradiction a sharp reviewer could find.

### Figures -- the big one (09:02 - 17:53)

This is where most of the meeting went, and Zaidi was blunt:

> "Now where this bloody figure one goes." [09:02]
> "Even I cannot really read [it]." [10:14]
> "These figures are so dim. I can't read any. How will the reviewer read?" [23:51]

The problem is mechanical: the figures were authored at full-page scale, and when dropped into an IEEE
half-column they become illegible. Zaidi's fix, which he performed live rather than described:

- Break the composite figure into its individual panels
- Re-lay them out **2x2** rather than 1x4 vertical, so each panel gets more area per column inch
- Increase the internal font size of each panel
- Add arrows and labels using Word's `Insert > Shapes` (visible in the Shape Format ribbon at [12:00])
- Set `Wrap Text > Square`
- Group everything into one object so it can be dragged as a unit
- Then, and only then, place it in a half column

He was explicit that this is an example to copy, not a finished product: "This is just an example I'm
giving." [17:07]

The panels visible on screen at [12:00] are the Kahlus architecture diagram: **A. Noninvasive Brain
Signals** (schematic electrode layout, "Past EEG window: 22 channels x 128 samples"), **B. Latent Neural
Field** (a 3D surface plot, "Hidden brain-state representation F_t ... inferred from observed signals,
not directly measurable"), and **C. Future Signal Prediction**.

### Cross-referencing (17:55 - 22:10)

None of the figures or tables are called in the main text. Zaidi caught all of them and inserted
"needs to be called in the main text" notes directly into the doc for Figure 1, Figure 2, Figure 3, and
Table 3.

He also found a numbering bug live: what the doc labels **Table 4** is actually Table 3, and there is no
real Table 4. Aayushya confirmed ("that's supposed to be table 3"). Table 3 also isn't legible and needs
the same picture-but-readable treatment as the figures.

### Results, claims, conclusions (22:24 - 25:33)

Zaidi read the results as: "this happened and it calculated 3.6" -- he registered the numbers, did not
interrogate them, and moved on to structure. What he wants:

- The supported / unsupported claims section fleshed out into an explicit claim-evidence mapping
- Conclusions and Discussion written (it is currently a bare heading)
- A decision on the sweep figures: either move them into Results **and** discuss them properly in
  Discussion, or drop them. Dumping unreadable, undiscussed plots near the references is not acceptable.

The sweep figures in question, visible at [24:00], are the honest ones -- panel title
"fair test: every learned model trained directly per horizon (lower is better)", plotting normalized
single-sample forecast MSE against forecast horizon (samples ahead @ 100 Hz), with five traces
(kahlus_gru, mean_of_trace, persistence, random, ridge_ar). This is the isolated non-overlapping result.
Zaidi's objection to them was purely that they are too small to read, not that they contradict the
headline.

### Housekeeping (26:00 - 28:41, 31:00 - 32:25)

- Filename stays `version 5 July 27`. Do not rename, do not restructure sections.
- New text in **red** so Zaidi can spot additions on his next pass.
- Draft in single column; he will handle final two-column layout.
- "Do not make any drastic changes on the formatting." [31:00]
- He will review again the following day: "If you do something tonight I'll check it tomorrow."

### Side threads (26:28 - 34:01, 32:25 - 37:17)

**IMAQI / Canada.** A separate abstract of Aayushya's was accepted at IMAQI, in Canada. Zaidi cannot
attend (full-time teaching) but may fly in for a Monday. This is the robot project, not this paper.
Aayushya noted the hardware had a setback -- a driver was destroyed by a power surge, replaced with a
new Teensy, stable again as of that day, with testing planned and a possible Institute visit Wednesday.

**Upcoming abstract deadlines.** SCUR is due **30 July 2026** (~250 words), IFORE sometime in August
(1500 characters). Both abstract-only. SCUR was three days out at the time of the meeting.

**Aayushya's second paper.** He reported killing his original direction -- tools/skills benchmarking for
agents -- after building it out and finding the data "very very not promising", plus existing prior work.
He has pivoted to: *can we experimentally identify which harness components cause changes in performance,
cost, and reliability?* Framed as: what would OpenAI or Anthropic have to change to optimize Claude Code
vs Codex specifically. Zaidi's reaction was immediate: "That is a hot topic. Right. It's very very hot."
Aayushya noted the frontier labs are focused on model quality rather than harness design, and that his
startup work is directly relevant ("we're essentially like a meta harness, we're able to create harnesses
from scratch"). He also floated an idea borrowing from the attention/transformer framing for this
application. Zaidi suggested Krish work with him to move faster.

**Zaidi's course.** He showed that he has enrolled in and paid for a 12-week UT Austin "Great Learning"
program in AI / agentic AI for business applications, and was looking at his own quizzes and projects.
"Become student again."

**Closing.** "You have to work like me. I'm working like crazy." Aayushya said he can commit 5-10 hours
a week to the research alongside a startup job that runs ~10 hours a day.

---

## Visual details from the recording

Frames sampled every 2 minutes at 1472 px wide.

**Setup.** Zaidi presented from a Windows machine (Windows 11 taskbar, clock bottom-right) via Zoom into
Aayushya's macOS. The Zoom banner "You are viewing syed zaidi's screen" is pinned top-center throughout.
Participant strip: Aayushya (camera on, glasses, dark room), syed zaidi (camera off, "S" avatar, green
active-speaker border almost continuously), Krish Garg (camera off, muted icon).

**Word.** Microsoft Word desktop, AutoSave **Off**, title `Paper - Aayushya - Krish - Zaidi - July 27-2026
version 5`. Font dropdown reads **Times New Roman** at 9-10 pt, so the Arial problem from version 4 has
been fixed. Word status bar tracks the doc growing and shrinking as figures move: page 1 of 10 / 3,227
words at 06:53 PM, page 4 of 11 / 3,224 words at 07:01 PM, page 6 of 11 at 07:07 PM, page 11 of 11 /
3,258 words at 07:13 PM. Still 10-11 pages against a 6-page limit.

**Confirmed fixed since version 4.** The sentence I flagged as truncated mid-word now reads complete:
"It is important to note the existing EEG models that are traditionally being used and develop prediction
models for EEG sensor data. This includes Ridge regression, covariance and Riemannian methods,
convolutional neural networks, recurrent neural networks, transformers, state-space models, and neural
translation models." Section II is properly titled "RELATED WORK AND MOTIVATION" and Section IV
"PROPOSED METHOD: KAHLUS-V1 NEURAL FIELD COMPILER".

**Figure 2 caption, in full, at [18:00]:**

> *Figure 2: Kahlus v1 Neural Field Compiler architecture. Input EEG (22 channels, 128 samples) is
> encoded into a latent neural field via a 12-layer GRU (384 dims), evolved causally, and decoded into a
> future EEG prediction. MSE = 3.116, Pearson r = 0.972.*

This is the invalid number, still in the paper, in a figure caption, unchallenged.

**Drive.** `drive.google.com/drive/u/0/folders/1c_js4EZfD9s-VneV6tp0un_iebe7MIK7`, path
`IEEE- IECBES2026- Mal... > Aayushya and Krish`, containing `archive/`, `kahlus_all_figures_2026-07-21/`,
`kahlus_amrith_forecast_figures_2026-07-21/`, all modified Jul 21, owner `aayu22809`.

**Zaidi's taskbar** showed a wide parallel workload: `Course Introduction`, two copies of the paper,
`Document1`, `Food Lab Meeting Jun...`, `Document2`, `Radhika IFORE Abstract`. His desktop (visible at
[34:00]) is dense with unrelated active projects -- `Plasma Woun...`, `Nitrogen Plasma`, `AI in
Parkinson`, `AI in Diabetes`, `Jellyfish_G`, `BARC3-2025`, `Illumina Sequencing`, `MATLAB R2023b`,
`Meta-Analysis Manual V4`, `SanJoseUni...`, `master_code`. Gmail showed 10,249 unread in one inbox and
`1 of 19,044`. Context for his throughput expectations.

---

## Full transcript

Auto-transcribed with faster-whisper (`small`, int8, CPU), 362 segments over 2,394 s, merged into ~40 s
paragraphs. Speaker labels are not separated -- the overwhelming majority is Zaidi; Aayushya and Krish
interject briefly. ASR errors are preserved rather than silently corrected; notable ones are "EG" for
EEG, "MRI" where MEG/fMRI is likely, "IntelliSceince"/"IntelliScience", "teen see" for Teensy, "scur"
for SCUR, "iPhone" for IFORE, "Timble" for Table, and "make the font read" for "make the font red".

**[00:00]** That's very good. So you are understanding how I have to look at those. So this is something that you have to do the way I'm doing it. Right? And try to put all these references in one file. So you go to Google, find the PDF file, save it somewhere for our reference. Because there is a possibility that these guys will come back to us and say, this reference is not accurate. So then we have to go back to our reference and we do not need to repeat our work again and again. So you just follow those. Okay. Now this is good. EG is attractive. So that's the way you have to do it. I don't want irrelevant references. They will immediately be destroyed. So make sure that you open that in Google or wherever and see if it is really relevant.

**[00:50]** Okay. EG is attractive and non-invasive. This is good. This is why EG is a difficult model. They are noisy, dimensional, different subjects. And then they have these problems such as this one, seven and eight. Now what is including front and center channels at seven and eight? I want to see what seven and eight are and why you have referenced them here. So what are seven and eight? So let's see seven. So what is seven? This is also a review. Right? Okay. Let's check. So this is a good paper.

**[01:34]** Right? So you can save this paper for deep learning of where did you get this? Where did you get this reference? I'll email you all the papers right now.

**[02:19]** Okay. So just make sure that these, what I'm trying to see is that we should be able to reach those papers and probably this one is the quality of life of patients. Deep learning. We know hyperlink the papers in the reference. There is? No, I'm asking can we do that? Yeah, you can. Yeah, that will be really great. Yes. Okay. I can go in and just try to see any that is not available. Then put some something else. Right? But make sure they are okay.

**[03:02]** Then you will say many machine learning techniques that intend to move this seven to nine. Now seven to nine must have eight and nine seven to nine seven and eight are included here. So you just probably put nine here. It's a seven eight already there in forecasting. This window makes the paper task. You will know the quality of this is okay. It is important to know that it is important that existing e-g models are traditionally being used and developed prediction models for a sense of data. These includes this, this, this, this, this, this, this, this, then I add that all these models are being extensively employed.

**[03:45]** Under various way in operating conditions with and without stimuli. However, there are certain limitations as these techniques are applied to the data. These limitations are described in section below providing motivation to conduct the research presented in the space. Right? That that that makes sense. Right. These limitations are asking us to do this. Now the related motivation. Okay. Existing modeling approaches are 10 to 15. So they tend to 15 must be related to the must have the CNN SS SMS EG self-supervised models. They must have those models.

**[04:31]** So make sure they are models. These are really that. Table one summarize the main modern families. So and their element relevance to this work, why it matters. This is good. So this is good. I like that table. So easy modeling approaches are significantly reporting and this is this is good. This is this is really good. Okay. This can produce decent predictions. Third, then fourth, right? Regression finally is this is good. This prompted the creation of one part inspired by this. So 16 should be trying me to.

**[05:18]** Right? I think it is. I saw that. Yeah, I think it is. I believe so. Because these guys will really check that. I tell you now. Yeah, I saw that. Yeah, here it is.

**[06:02]** That was a really good. They have 1000 hours of MRI across seven to be subjects. And they call it try me to. Yeah, this is good. This is really good. But that we have. We have a correct reference. Okay, this is good. So therefore we do says 16, where was 16 here? Okay, that is good. So is this 17 is regression? 17 regression. Very good.

**[06:49]** Very good. This is really good. So make sure that 18 and 19 are correct. Very good. This is objectives. So research objectives are again, you are giving one to three. I don't know why one to three are here. So you have to check that. Do we really need it? No, I should be somewhere else. So in the current paper, we validate the first practical part of the framework inspired by prior work in neuroscience and space and model and observation.

**[07:41]** So make sure that these, these guys at 1221, 2223, they are correct. Okay. Just to perform. Now it is one and three is where is the where is this more why it is the third reference is the mollab paper. Okay. The more you put them more difficult you make. So these are the research objectives.

**[08:23]** Can you read the, can you both read the research objectives is correct. Yeah. The objectives of the proposed work were to provide a clear future e.g. signal forecasting tasks definition for the moab data set to provide a proposed neuro field compiler framing that treats e.g. As an observation of like a hidden brain state field and to provide a comparison against ridge regression on the same forecasting task and to develop evidence gates that keep the paper from claiming more than what the data actually supports. So these are the ones which we have worked right. That's good. Now where this bloody figure one goes. What is this figure and I can't put it.

**[09:08]** You know your problem. Your problem is that when I even put it like this for a single column, this is what you will see. That is not right. Right. So what you have to do is you have to make. You have to make this in a way that they can be they can be fit. So you have to make it for I will make it for you. I'll give you a screen. So I will just say here, for example, when I can't read now what is in the bottom. And I can take it here. And I have to make sure that I make really this right and then I will go and then I will cut this or you do whatever you want and you cut this right.

**[10:06]** This is not right. So that's good. So even I cannot really read. So you have to read. You see the at the bottom is something is written. So you have to read it properly. Right. And then you go on the third one, which is so you have to remake them. They cannot fit. And if they cannot fit, they will really, you know, they will just they will just and and then you can put. The last one.

**[11:08]** And now what you can do is you can make a right. So you can make one, two, three, four. So now you can make insert and shape line. They put from here. You make it anyway like this. Okay.

**[12:59]** And make it the way that this the way I do this. Okay. So first of all, you make them wrap text square. Make it right. Probably. Combine all of it.

**[13:41]** Hold on. So you write in a way that it comes in a column. In like the tower. Yeah. Yeah. And it comes in a column. Then what can I do later? But you see it can't read anything. So you make the forms because then later on what we can do is we can basically add all of it. Right. And I will say that we will just as a picture. Now, this is basically a picture. And if I can put a grab text and they put this type of a movie here and there and I really do not need those.

**[14:29]** And I delete all of them. Now this guy I can move. Right. So this is what I need. And then I will write here if you have one, but they should I can make it bigger. This as well. And I can also add it here. You see that that I can write it something and I can put it wherever I want. But that's okay. You make the picture like this. So this guy is done. Right. And now you have to write here figure one. I don't know where this figure will go. So you have to you have to put somewhere where this figure one is going and put that put the title here as well.

**[15:21]** Okay. So this is this is what you have to see. Okay. Now, next, the data source. So this is what it has to be more clear and more. I have just given you an example. How should I do and this should be readable and very nice. Now, data source and preliminary analysis. Okay. Present study. Okay. So I will put it only three. Because that's what I need this public data set as this, this, this, the materials, easy signals are there. This is more raw data. Subject is okay. This probably it's okay.

**[16:05]** If you increase the font size, what is written in the bottom. Because if I make it smaller, you know, for a half column, look, you can't read anything. Right. When you read anything, this is half the column, right. You can't read anything. So why we are putting it this. So you have to read some intelligent way. So do two of them. Then I don't have that. Right. And then to that will be really great. What I'm saying is you can intelligently make it. You can intelligently make it like this.

**[16:49]** Two of them. Right. So you can say two of them. Right. And then you can increase the size. And then the rest of the two of them. And then you can put arrows and you can write there whatever you want. This is just an example I'm giving. Here. Right. So the, it gets to fit. Really. And then I can combine those in a group. I can say here. I can just say, let's say square. And then I will just say. Let's say. Square.

**[17:31]** And then I will just put it here. And I will just get rid of. Here. And then you just combine them as a group. So now this is the picture, which is in the half. All right. In half column. It fit in half. And you're right here what the pictures. So this is crazy. can put your figure 2 there whatever the difference. Now I do not know where figure 1 is called. So this figure 1 needs to be called in the main text. So I will write here needs to be called in the main text. So I don't know where we are. Now figure 2 should also be called somewhere. I don't see any place where you are calling figure 2. Why it is there?

**[18:28]** So it needs to be called which is important. So I will just write here for you that figure 1. So here is the figure which needs to be called in the text figure 2. So that is good. Then you go for proposed method. So this is okay. This is very nice. I made this. I made

**[19:17]** this value so that we can fit it in half column. This is good. 1 is good. Timble 2. Timble 2 is fine. I made it small and I made it as a picture so I can move it around. This is good. Timble 2 you do not need to do anything. This is okay. Then you have oh my god. Then you have table 2 and table 2 is okay. Then you have experiment to put upon which is okay. Okay. So this guy

**[20:16]** has to be done exactly the way I did the part. Right? Like where is it? And figure 3 needs to be called somewhere. So I don't know where is figure 3 and I will just say it needs to be called. Okay. And then this is fine. Results. Now these results are there but then this is good. But then table 3 needs to be called. Very stable 3. So I couldn't find table 3. I read it but I was not able to find table 3. So table 3 needs to be called

**[21:14]** somewhere. Did it get removed? Where is that? Now table 3? Yeah table 3. Where are you calling it? Right there. It should have been integrated with the diagnostics. Table 4 is I don't know which table 4 is. I don't have any table 4. Is it this one? Oh this is table 4. Is it 3? Is it this one? Summarize it. This table is integrated. Oh yeah that's supposed to be table 3. This

**[22:04]** should be table 3. Table 3. Yeah. The claim is in the recorded evidence. Yeah so you see that table 3 you have to make as a picture but make sure that it's readable. Table 3 is not readable. Right. Now so results are only these ones. These are the results where you are saying that this happened and it calculated 3.6. This one persisted and this one baseline and this one is not. So now the question is supported and unsupported claims. You have to write a little bit more on it. Right. The claim interpretation from the current evidence.

**[22:53]** What is that? So you have to write down that claim evidence. Right. Needs to complete. Okay. Then you have conclusion and discussion is nothing there. So I will just say needs to complete. It's actually. And then future work is here. The references. I don't think that we should really put those these pictures here. Okay. I don't think that we should add them but if you want to add them then you have to put them in results. Right. They will come in results. Right. Yeah. So I will say that. Okay. So hold on. Okay. So I will say

**[23:51]** that these pictures. All these figures. We see these figures are so dim. I can't say read any. I can't read any. How the reviewer will read. So if you want to add these figures in the section then we really also add the results section also add discussion results in the discussion section. So this is what you have to do. If you want they are good

**[25:15]** but I can't read anything so I don't know what. Right. Even if I make it full page I can't read. I don't know what it represents. I don't know what they are. So this is nothing really there. So you have to your figures are just not reading. And then you put those. This is important what I have written there. So I think we are working on it and I think it will be a great paper but you have to make it properly for the people. I will save it and I will now do not change the name. Just add everything there whatever it is. Okay. And do not feel sections which are already there. And if you add

**[26:09]** something new make it read. So that when I read I say oh that one is the one which you have added and make all these figures. There is a lot of work to be done. Okay. But we are almost there. We are almost there and it will be accepted. I assure you by the way your I think you're one of the one of the abstract is accepted in IMAQI. Yeah. I'm a key got accepted. So we're going to Canada. I don't know you will go to Canada. I can't because I will have full time teaching so I really cannot go but I will try to come at least for Monday. Yeah. But that will be really great to join that huge conference.

**[27:02]** Which paper would abstract this. That was IMAQI paper. That was the robot project. So not this one. Yeah. That's whatever he's making which has not worked yet. It has to work. It has to it has to carry my has to carry my torch. I don't know how far you are now. How far you are. So I'm going to do testing today. Yesterday what happened was one of the drivers got flamed because there was a power surge. And so I got a new teen see and I was checking out so everything looks stable today. So I have it set up right there where I'll test again today if it works then I'll come in I think Wednesday in the Institute and then yeah.

**[27:54]** So I believe that I will upload this paper. It's called version 5 July 27. Don't make those actions. Just make one section. Just write it as single single column and keep writing and whatever you will add. Make the font read so I know that this new edition has been done. Okay. This is very important. This is a great paper and I believe I really believe that it will be. It will be it will be accepted. That's what I feel. IntelliScience conferences and where we are sending it. I see be Malaysia one. Yeah, there is version 5 July 27. Okay.

**[30:46]** Okay. Yeah. So do not make any drastic changes on it. Okay. Yeah. Don't make any drastic changes on the formatting. Just make sure that you just nicely put together everything and whatever. So you do not have much to work really there is figures what you have to make and make sure that these figures are called in the big figures are called you know these people are called in the main text and you read those figures and this figure is made called so you also yeah so this is okay. So you have to remake this one and need to be called in the way in text. Results are okay. Yeah, so you can put it in the next one. That's okay. So you need to complete the claims you have to put properly and then it needs to complete the

**[31:58]** discussion and in results obviously if you want these graphs then you have to look at where to put them and how to discuss them. Okay. All right. Okay. Awesome. Yeah. Yeah. I'm working continuously on it. This is a really great paper. I want to submit it. Please work on it. I will work on it tomorrow as well. If you do something tonight I'll check it tomorrow. Okay. Okay. Stay on for a second for me real quick. Yeah. So I didn't I know we haven't talked about the paper that I was writing separate from my usual. Yeah. So just a quick update on it. I was initially trying to do the I don't know if you remember but tools and skills benchmarking. Yes. That's why people need us for agents. But after like some building it out and testing getting some data it was very very not promising. It seems like there's also some work done around it. So I kind of I've changed it and doesn't work around the question.

**[33:07]** Can we experimentally identify which harness components caused changes in performance cost and reliability. So for example if any user using cloud code versus codex what is it that open AI or anthropic has to change to optimize each individual. That is a hot topic. Right. It's very very hot. Yeah. No work done even even the frontier labs haven't touched this at all. They're very focused on just making the models themselves better. So I've been I've been doing a lot of work on this. Yeah. Do that. And you will be but we have to submit scur and iPhone both abstracts on abstracts only you said. Yeah. These two conferences need abstracts only one to 50 words one one one abstract the other abstract should be 1500 characters. What's the deadline for that. The deadline for is 30th July for for first and for iPhone you can submit it in August sometime. Okay. Let me let me try to see if I can speed this up.

**[34:19]** Yeah. Speed it up. Twist the hand of this guy. Yeah. You should do you want to help me with this. It might go faster. Just ask. Okay. Ask him to work with you and I will ask it to work with you. Okay. I think I think it's just been a little harder with it seems like you should know more about this. This idea I actually had an idea to do something similar to the attention framework and transformer for this application. And I had an idea. Okay. Submit I for friends. Yeah.

**[35:02]** I was also doing a show you that here. So Fish. I just have So this is the course. I have been now I just

**[35:57]** Registered myself For this course. Great learning. I think it program in AI, Agency and the way I Business application. This is it with the UT Austin. So I just picked the fee for 12 weeks. And now I was looking at my quizzes and projects. Become student again. Yeah. This is interesting stuff. I've been so at the startup that I'm working at as an engineer. We're essentially like a meta harness. We're able to create harnesses from scratch. I see. So it's like very close to the work I'm doing there, which is why there's a lot of expertise. Yeah.

**[36:41]** Okay, do that. And then I think we will be able to finish that one. Okay. Yeah. Yeah. I'll update the drive so you can stay up to date. But Participating conferences and And good conferences. You have to push the work as much as you can. Yeah. You have to work like me. I'm working like crazy. Yeah, I know. The start of this is the only way to do it. I'm only going to be devoting five, 10 hours a week on the research. Excellent. Start of is like 10 hours a day. It's pretty, it's pretty brutal. But um, Yeah. Okay. All right. Thank you very much. Thank you.

**[37:30]** Thank you. Yeah. Thank you.

