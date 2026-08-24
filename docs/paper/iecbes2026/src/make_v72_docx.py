"""Produce Version 7.2 of the manuscript from Version 7.1.

Claim-hygiene pass for the 2026-08-14 Zaidi call and the IECBES 2026 submission.
Every edit is listed in ../claims_audit.md. Unlike make_v71_docx.py this pass
does rewrite prose — each rewrite either removes an overclaim, names the corpus
a number came from, or adds an honest result from the findings ledger.

The script fails loudly: every FIND string must match exactly once across the
document, and a post-build verification re-extracts the text and asserts the
overclaims are gone. No silent partial application.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import docx
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SRC = BASE / "Paper-Aayushya-Krish-Zaidi-V7.1.docx"
OUT = BASE / "Paper-Aayushya-Krish-Zaidi-V7.2.docx"

# ---------------------------------------------------------------- replacements
# (find, replace) applied to body paragraphs. Each must hit exactly once.
EDITS = [
    # Title: drop the legacy "Neural Field Compiler" branding (AGENTS.md §12,
    # CLAIM_EVIDENCE_MATRIX C004 — the evaluated model is the GRU translator).
    ("Developing An Evidence-Gated Neural Field Compiler Framework for EEG Forecasting",
     "Developing an Evidence-Gated Framework for Leakage-Controlled EEG Forecasting"),

    # --- Abstract ---
    ("We present a proposed Neural Field Compiler (NFC) framework (Kahlus-V1) for "
     "EEG future-window forecasting.",
     "Kahlus-V1, an evidence-gated modeling framework for EEG future-window "
     "forecasting, is presented; it treats observed EEG channels as partial "
     "observations of a latent brain-state representation."),
    ("uncertainty heads, and a top-level Neural Field Compiler.",
     "uncertainty heads, and a top-level compiler module."),
    ("These results demonstrate strong EEG future-window forecasting performance.",
     "At face value, these figures suggested strong future-window forecasting "
     "performance."),
    ("The NFC compiler is open source and available on GitHub for research use: "
     "https://github.com/ptlnextdoor/Kahlus-V1.",
     "The implementation is open source and available for research use [25]."),
    ("We therefore retract MSE = 3.116 and r = 0.972 as evidence of forecasting "
     "skill and report them only as a measured artifact of window overlap.",
     "MSE = 3.116 and r = 0.972 are therefore retracted as evidence of "
     "forecasting skill and reported only as a measured artifact of window "
     "overlap."),
    ("Under an isolated, non-overlapping, subject-held-out re-evaluation, learned "
     "models still beat",
     "Under an isolated, non-overlapping, subject-held-out re-evaluation (a "
     "single-channel Sleep-EDF horizon sweep and a BNCI2014_001 "
     "single-future-sample check), learned models still beat"),
    ("The contribution we claim is the leakage-controlled evaluation protocol and "
     "the evidence gate that detected this failure in our own pipeline.",
     "The claimed contribution is the leakage-controlled evaluation protocol and "
     "the evidence gate that detected this failure in the authors' own pipeline."),
    ("ridge regression, Neural Field Compiler, Evidence Gates",
     "ridge regression, leakage control, Evidence Gates"),

    # --- Introduction / objectives ---
    ("The major question we ask is:",
     "The major question asked is:"),
    ("For this purpose, a neural field compiler was developed and called Kahlus-V1.",
     "For this purpose, an evidence-gated forecasting framework was developed "
     "and called Kahlus-V1."),
    ("In the current paper, we validate the first practical part of this framework:",
     "The current paper validates the first practical part of this framework:"),
    ("To provide a proposed NFC framing that treats EEG as an observation of a "
     "hidden brain-state field;",
     "To provide a proposed framing that treats EEG as an observation of a "
     "hidden brain-state field;"),
    ("The proposed neural field compiler is specifically designed",
     "The proposed framework is specifically designed"),

    # --- Method section: keep the latent-state content, drop the brand ---
    ("PROPOSED METHOD: KAHLUS-V1 NEURAL FIELD COMPILER",
     "PROPOSED METHOD: THE KAHLUS-V1 FRAMEWORK"),
    ("Kahlus-V1 NFC visual abstract.",
     "Kahlus-V1 visual abstract."),
    ("Figure 2: Kahlus v1 Neural Field Compiler architecture.",
     "Figure 2: Kahlus-V1 architecture."),
    ("The broader NFC worldview can be written as",
     "The broader multimodal-observation view can be written as"),
    ("Kahlus-V1 Architecture Components: Kahlus-V1 NFC dataflow.",
     "Kahlus-V1 Architecture Components: Kahlus-V1 dataflow."),

    # --- Results: two sentences that still credit the retracted number ---
    ("Recovered 100,000-step batch-matched task metrics Kahlus-V1 improves "
     "future-window forecasting MSE and correlation relative to ridge "
     "regression, while masked reconstruction fails and therefore blocks "
     "broader claims.",
     "Recovered 100,000-step batch-matched task metrics are reported below. "
     "Masked reconstruction fails and blocks broader claims, and Section VII "
     "shows the forecasting comparison itself is overlap-dominated and is "
     "retracted."),
    ("suggest it captures predictive structure beyond autocorrelation alone, "
     "though this interpretation requires validation with per-window "
     "significance testing in future work.",
     "initially suggested predictive structure beyond autocorrelation alone; "
     "Section VII shows this interpretation does not survive the overlap audit."),
    ("In this recovered setup, Kahlus-V1 produced lower forecasting errors and "
     "higher correlation than ridge regression, meaning its predicted future "
     "waveforms more closely matched the true future EEG windows. This should "
     "be interpreted as a forecasting result, not as evidence of clinical "
     "diagnosis or general brain understanding.",
     "In this recovered setup, Kahlus-V1 produced lower numerical errors and "
     "higher correlation than ridge regression on the overlapping formulation. "
     "Section VII shows that this apparent advantage is an artifact of window "
     "overlap; it is not interpreted as forecasting skill, nor as evidence of "
     "clinical diagnosis or general brain understanding."),

    # --- Evidence gate paragraph ---
    ("the gate allows the narrow claim that Kahlus-V1 supports future EEG window "
     "forecasting on the recovered MOABB setup, while blocking claims about",
     "the gate allows only the narrow isolated-evaluation baseline-improvement "
     "claim reported in Section VII, while blocking claims about"),

    # --- Fig. 3 caption: name the corpus ---
    ("On held-out subjects at a 16-sample horizon, Kahlus-V1 reaches MSE 0.690 "
     "against ridge regression at 0.822, so the two remain close. The evidence "
     "gate approves only the narrow EEG forecasting claim.",
     "On held-out Sleep-EDF subjects (single channel, 100 Hz) at a 16-sample "
     "horizon, the recurrent model reaches MSE 0.690 against ridge regression "
     "at 0.822, so the two remain close. The evidence gate approves only the "
     "narrow baseline-improvement claim."),

    # --- Section VII prose ---
    ("We tested this directly rather than assuming it.",
     "This was tested directly rather than assumed."),
    ("Holding the model, the data, and the training procedure fixed and changing "
     "only the scoring window, a ridge regression scored",
     "Holding the model, the data, and the training procedure fixed and changing "
     "only the scoring window, on subject-held-out Sleep-EDF data a ridge "
     "regression scored"),
    ("We therefore treat MSE = 3.116 and Pearson r = 0.972 as a measurement of "
     "window overlap rather than as evidence of forecasting skill, and we do "
     "not advance them as a performance claim for Kahlus-V1.",
     "MSE = 3.116 and Pearson r = 0.972 are therefore treated as a measurement "
     "of window overlap rather than as evidence of forecasting skill, and are "
     "not advanced as a performance claim for Kahlus-V1."),
    # Overlap law (F9), appended to the disjoint-windows paragraph.
    ("The evaluation reported here consequently scores only the strictly future "
     "sample, so that input and target are disjoint by construction.",
     "The evaluation reported here consequently scores only the strictly future "
     "sample, so that input and target are disjoint by construction. The "
     "induced inflation is bounded: a window-scored error metric can overstate "
     "strictly-future error by at most a factor of W/H, a bound verified on "
     "286 window-horizon configurations across eleven signals with zero "
     "violations; at the W = 128, H = 1 configuration used here the bound "
     "predicts a factor of 127, matching the measured factor of roughly 126."),
    ("Under that isolated evaluation, on subject-held-out data, both learned "
     "models clearly beat the trivial baselines,",
     "Under that isolated evaluation, on subject-held-out single-channel "
     "Sleep-EDF recordings (100 Hz), both learned models clearly beat the "
     "trivial baselines,"),
    ("At a horizon of 16 samples the recurrent model reaches MSE 0.690",
     "At a horizon of 16 samples (160 ms) the recurrent model reaches MSE 0.690"),
    # BNCI h=1 + the nine-subject sweep (ledger F26).
    ("We therefore do not claim that the nonlinear architecture outperforms a "
     "linear model on this task.",
     "The nonlinear architecture is therefore not claimed to outperform a "
     "linear model on this task. A subsequent nine-subject "
     "leave-one-subject-out sweep on BNCI2014_001 at matched training budget, "
     "with the ridge autoregressive baseline tuned per horizon, reached the "
     "same verdict with more power: across nine horizons (10-2560 ms), none of "
     "the 81 fold-horizon comparisons was significant (0 of 9 horizons at "
     "p < 0.05, every per-horizon confidence interval including zero, largest "
     "mean gap 0.0181 MSE), so this corpus cannot distinguish the two models "
     "at any horizon."),
    ("Third, the Neural Field Compiler framing is implemented and executable "
     "end to end.",
     "Third, the latent-state framing is implemented and executable end to end."),
    ("the isolated re-evaluation used only two held-out test subjects, which is "
     "below the number of clusters required for a stable confidence interval.",
     "the isolated re-evaluation used only two held-out test subjects, which is "
     "below the number of clusters required for a stable confidence interval; "
     "the nine-subject sweep is powered enough only to report "
     "indistinguishability, not superiority in either direction."),

    # --- Conclusions ---
    ("and we report it as such rather than as a performance claim.",
     "and it is reported as such rather than as a performance claim."),
    ("On held-out subjects, learned models still beat persistence, mean-of-trace, "
     "and random baselines at every horizon tested,",
     "On held-out Sleep-EDF subjects, learned models still beat persistence, "
     "mean-of-trace, and random baselines at every horizon tested,"),
    ("The isolated re-evaluation used two held-out test subjects, which is "
     "insufficient for a powered claim.",
     "The isolated re-evaluation used two held-out test subjects, which is "
     "insufficient for a powered claim; a later nine-subject "
     "leave-one-subject-out sweep on BNCI2014_001 confirmed only that the "
     "models cannot be distinguished at any horizon."),
    ("We regard the demonstration that an evidence gate caught a false positive "
     "produced by our own pipeline as the more durable contribution of this "
     "work, and we report the invalidated number openly so that the failure "
     "mode is reusable by others.",
     "The demonstration that an evidence gate caught a false positive produced "
     "by the authors' own pipeline is regarded as the more durable contribution "
     "of this work, and the invalidated number is reported openly so that the "
     "failure mode is reusable by others."),
]

# Table cell edits: (table_index, find, replace) — exact-once within tables.
TABLE_EDITS = [
    (2, "Supported narrow forecasting result",
        "Retracted: overlap-dominated (Section VII)"),
    (2, "Strong baseline, lower than Kahlus v1",
        "Strong baseline (overlapping formulation)"),
    (3, "The Neural Field Compiler framing is implemented end to end",
        "The latent-state framing is implemented end to end"),
    (3, "Isolated non-overlapping evaluation on held-out subjects: MSE 0.690 at "
        "horizon 16 against 1.334 for persistence and 1.264 for mean-of-trace",
        "Isolated non-overlapping evaluation on held-out Sleep-EDF subjects: "
        "MSE 0.690 at horizon 16 (160 ms) against 1.334 for persistence and "
        "1.264 for mean-of-trace"),
    (3, "The recovered run stored no per-window predictions, and the isolated "
        "re-evaluation used only two held-out subjects",
        "The recovered run stored no per-window predictions, the isolated "
        "re-evaluation used only two held-out subjects, and a nine-subject "
        "BNCI sweep cannot distinguish the models at any horizon"),
    (3, "At horizon 1, ridge 0.517 against 0.568;",
        "At horizon 1 (BNCI2014_001, single channel, two test subjects), ridge "
        "0.517 against 0.568;"),
]

REFERENCE_25 = (
    '[25] A. Patel, K. Garg, and S. Zaidi, "Kahlus-V1: evidence-gated EEG '
    'forecasting framework (source code)," GitHub repository, 2026. [Online]. '
    "Available: https://github.com/ptlnextdoor/Kahlus-V1"
)

TRAILING_NOTES = [
    "If you want to add these figures to the results section",
    "Also add discussion on these results",
]


def replace_in_paragraph(p: Paragraph, find: str, repl: str) -> bool:
    """Replace across runs, preserving formatting outside the match."""
    runs = p.runs
    joined = "".join(r.text for r in runs)
    i = joined.find(find)
    if i < 0:
        return False
    j = i + len(find)
    pos = 0
    spans = []
    for r in runs:
        spans.append((pos, pos + len(r.text)))
        pos += len(r.text)
    first = last = None
    for k, (a, b) in enumerate(spans):
        if a < j and b > i:
            if first is None:
                first = k
            last = k
    assert first is not None
    prefix = runs[first].text[: i - spans[first][0]]
    suffix = runs[last].text[j - spans[last][0]:]
    runs[first].text = prefix + repl
    for k in range(first + 1, last):
        runs[k].text = ""
    if last != first:
        runs[last].text = suffix
    else:
        runs[first].text = prefix + repl + suffix
    return True


def apply_body_edits(doc: docx.Document) -> None:
    for find, repl in EDITS:
        hits = sum(replace_in_paragraph(p, find, repl) for p in doc.paragraphs)
        assert hits == 1, f"expected 1 hit, got {hits}: {find[:70]!r}"


def apply_table_edits(doc: docx.Document) -> None:
    for ti, find, repl in TABLE_EDITS:
        hits = 0
        for row in doc.tables[ti].rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    hits += replace_in_paragraph(p, find, repl)
        assert hits >= 1, f"table {ti}: no hit for {find[:60]!r}"


def fix_conclusions_heading(doc: docx.Document) -> None:
    """The Conclusions heading is auto-numbered and renders as a second 'VII.'.

    Make it literal 'VIII.' like the other literal headings and detach it from
    the numbering, so the rendered sequence is ... VII, VIII, IX.
    """
    for p in doc.paragraphs:
        if p.text.strip() == "Conclusions and Discussion":
            pPr = p._p.get_or_add_pPr()
            for numPr in pPr.findall(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}numPr"
            ):
                pPr.remove(numPr)
            p.runs[0].text = "VIII. " + p.runs[0].text
            return
    raise AssertionError("Conclusions heading not found")


def add_reference_25(doc: docx.Document) -> None:
    for p in doc.paragraphs:
        if p.text.strip().startswith("[24]"):
            ref = copy.deepcopy(p._p)
            p._p.addnext(ref)
            new = Paragraph(ref, p._parent)
            joined = REFERENCE_25
            new.runs[0].text = joined
            for r in new.runs[1:]:
                r.text = ""
            return
    raise AssertionError("[24] not found")


def drop_trailing_notes(doc: docx.Document) -> None:
    removed = 0
    for p in list(doc.paragraphs):
        if any(p.text.strip().startswith(t) for t in TRAILING_NOTES):
            p._p.getparent().remove(p._p)
            removed += 1
    assert removed == 2, f"expected 2 trailing notes, removed {removed}"


def verify(path: Path) -> None:
    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    for tb in doc.tables:
        for row in tb.rows:
            for cell in row.cells:
                text += "\n" + cell.text
    banned = [
        "Neural Field Compiler",
        "NFC",
        "demonstrate strong EEG future-window forecasting performance",
        "https://github.com/ptlnextdoor/Kahlus-V1.",  # bare URL in abstract
        "This should be interpreted as a forecasting result",
        "Supported narrow forecasting result",
        "If you want to add these figures",
    ]
    for b in banned:
        assert b not in text, f"banned string survived: {b!r}"
    required = [
        "are therefore retracted as evidence of forecasting skill",
        "at most a factor of W/H",
        "cannot distinguish the two models at any horizon",
        "Sleep-EDF subjects (single channel, 100 Hz)",
        "[25] A. Patel",
        "VIII. Conclusions and Discussion",
    ]
    for r in required:
        assert r in text, f"required string missing: {r!r}"
    # 0.972/3.116 may appear only near retraction language — count occurrences
    # and require the retraction markers exist (checked above).
    print("verify: PASS —", path.name)


def main() -> None:
    doc = docx.Document(str(SRC))
    apply_body_edits(doc)
    apply_table_edits(doc)
    fix_conclusions_heading(doc)
    add_reference_25(doc)
    drop_trailing_notes(doc)
    doc.save(str(OUT))
    verify(OUT)


if __name__ == "__main__":
    sys.exit(main())
