#!/usr/bin/env python3
"""Check every quantitative claim in the primer against its committed artifact.

The primer is what a non-specialist reads to learn what this project found, so a
stale number in it is worse than one in a draft: nobody reading it is in a
position to notice.

Design note, learned the hard way. The first version of this script asked "does
the string 0.690 appear in the document?" That is nearly worthless, because most
numbers appear two or three times (a results sentence, the summary Q&A, the
glossary). Corrupting one site left the others intact and the check still
passed -- a planted-fault self-test caught 2 of 7 faults.

So each check here *extracts* the number from a regex anchored to its own
sentence and compares that captured value to the artifact. Every site of a
repeated claim is anchored separately, which additionally catches the case where
the results section and the summary disagree with each other.

Usage:
    python3 docs/paper/primer/verify_primer.py         # verify
    python3 docs/paper/primer/verify_primer.py --self-test   # prove it can fail
Exit status is non-zero on any failure, so it can gate a commit.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEX = ROOT / "docs/paper/primer/kahlus_primer.tex"

ENVELOPE = ROOT / "artifacts/envelope_sleepedf/results.json"
STAGE = ROOT / "artifacts/stage_forecast/transition_detection_n1_stratified.json"
REGISTER = ROOT / "docs/research/program/REGISTER.md"
CORRELATION = ROOT / "docs/results/correlation_inflation_law_2026-07-30.md"

NUM = r"([0-9]*\.?[0-9]+)"


class Checker:
    def __init__(self, tex: str) -> None:
        self.tex = tex
        #: LaTeX rewraps paragraphs, so a sentence anchor must not depend on
        #: where the line breaks fell. Match against a whitespace-flattened
        #: copy and write patterns with plain spaces.
        self.flat = re.sub(r"\s+", " ", tex)
        self.failures: list[str] = []
        self.n = 0

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        self.n += 1
        if not ok:
            self.failures.append(label)
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))

    def anchored(self, label: str, pattern: str, expected: float,
                 tol: float = 5e-4, scale: float = 1.0) -> None:
        """Extract a number from `pattern` in the tex and compare to the artifact.

        A missing anchor is a failure, not a skip: if the sentence was reworded
        the check has stopped guarding anything and needs updating.
        """
        m = re.search(pattern, self.flat)
        if not m:
            self.check(label, False, "anchor not found; sentence reworded?")
            return
        got = float(m.group(1)) * scale
        self.check(label, abs(got - expected) < tol,
                   f"tex={m.group(1)} artifact={expected / scale:.5g}")

    def present(self, label: str, needle: str, want: bool = True) -> None:
        self.check(label, (needle in self.tex) == want, f"'{needle}'")


def build(tex: str) -> Checker:
    c = Checker(tex)

    # ---- F11: envelope forecasting ------------------------------------------
    env = json.loads(ENVELOPE.read_text())
    lg = env["bands"]["low_gamma"]["horizons"][-1]
    c.check("F11 longest horizon is 60 s", lg["horizon_s"] == 60.0, f"{lg['horizon_s']} s")

    r60 = lg["eeg"]["pearsonr"]
    r_res = lg["eeg_artifact_residualised"]["pearsonr"]
    # Quoted twice: once as the headline result, once in the artifact-control
    # sentence. Anchor both, so a half-updated edit cannot slip through.
    c.anchored("F11 low-gamma r at 60 s (result sentence)",
               r"low-gamma envelope correlation is \$r = " + NUM, r60)
    c.anchored("F11 low-gamma r at 60 s (control sentence)",
               r"60\\,s low-gamma falls from \$r = " + NUM, r60)
    c.anchored("F11 r after EMG/EOG residualisation",
               r"falls from \$r = [0-9.]+\$ to \$r = " + NUM, r_res)

    # "every horizon, every band, all subjects" is universally quantified, so it
    # must be checked over every cell rather than the one that got quoted.
    all_beat, n_test = True, set()
    for bd in env["bands"].values():
        for h in bd["horizons"]:
            e = h["eeg"]
            n = e["paired_vs_persistence"]["n"]
            n_test.add(n)
            all_beat &= e["beats_persistence_n"] == n
    c.check("F11 every band x horizon: all 15 test subjects beat persistence",
            all_beat and n_test == {15}, f"n_test={sorted(n_test)}")
    c.anchored("F11 '15 of 15' matches artifact (results)",
               NUM + r" of 15 held-out test subjects beat persistence", 15.0, tol=0.5)
    c.anchored("F11 '15 of 15' matches artifact (summary)",
               r"to 60\\,s and beyond, on " + NUM + r" of 15 held-out", 15.0, tol=0.5)

    # ---- F12: stage and transition forecasting ------------------------------
    st = json.loads(STAGE.read_text())
    by = st["transition_detection"]["by_input"]

    def at(key: str, horizon: float = 60.0) -> dict:
        return next(r for r in by[key] if r["horizon_s"] == horizon)

    so, ep, ap = at("stage_only"), at("eeg_plus_stage"), at("all_plus_stage")

    c.anchored("F12 AUC stage only (results)",
               r"ROC curve rises \\textbf\{" + NUM + r"\} \(stage alone\)", so["roc_auc"])
    c.anchored("F12 AUC + EEG (results)",
               r"\\textbf\{" + NUM + r"\} \(with EEG\)", ep["roc_auc"])
    c.anchored("F12 AUC + all PSG (results)",
               r"\\textbf\{" + NUM + r"\} \(all channels\)", ap["roc_auc"])
    # Same pair restated in the Q&A; disagreement between the two is the bug.
    c.anchored("F12 AUC stage only (summary)",
               r"EEG does add: AUC " + NUM, so["roc_auc"])
    c.anchored("F12 AUC + EEG (summary)",
               r"EEG does add: AUC [0-9.]+\s*\n?\$\\rightarrow\$ " + NUM, ep["roc_auc"])

    c.anchored("F12 average precision, stage only",
               r"average precision \\textbf\{" + NUM, so["average_precision"])
    c.anchored("F12 average precision, + EEG",
               r"average precision \\textbf\{[0-9.]+\} \$\\rightarrow\$\s*\n?\\textbf\{" + NUM,
               ep["average_precision"])
    c.anchored("F12 average precision, + all PSG",
               r"\\textbf\{" + NUM + r"\} against a base rate", ap["average_precision"])
    c.anchored("F12 base rate", r"against a base rate of \\textbf\{" + NUM + r"\}",
               so["base_rate"])

    # The N1 share is horizon-dependent. Quoting the 10 s value in a sentence
    # about 60 s is the exact error this file was written to catch.
    # The primer now makes one rounded claim ("just under 58 %") that has to
    # hold at *every* horizon, not just the one that happened to be quoted.
    # Checking the max keeps it falsifiable: if any horizon exceeded 58 %, the
    # word "under" would be wrong.
    n1_shares = [r["n1_share_of_transitions"] * 100 for r in by["stage_only"]]
    c.anchored("F12 N1 share (primer states about 58 %)",
               r"\\textbf\{" + NUM + r"\\,\\%\} of transitions involve stage N1",
               58.0, tol=0.01)
    # "About 58" has to be true at every horizon the sentence covers, not just
    # the one that happened to get quoted. The first draft said "just under 58"
    # and this check failed it: the 60 s horizon is 58.07 %.
    c.check("F12 N1 share rounds to 58 % at every horizon",
            all(57.5 <= v_ < 58.5 for v_ in n1_shares),
            " ".join(f"{v_:.2f}" for v_ in n1_shares))

    # The 10-subject run said the effect "largely collapses" without N1; at 60
    # subjects it is the reverse, so these anchors moved to the new sentence.
    # All three horizons are quoted, so all three are checked.
    for i, h in enumerate((10.0, 30.0, 60.0)):
        pre = r"chance\} --- AUC \\textbf\{" + (r"[0-9.]+ / " * i)
        c.anchored(f"F12 AUC excluding N1, stage only, h={h:.0f}s",
                   pre + NUM,
                   at("stage_only", h)["excluding_n1_transitions"]["roc_auc"])
    for i, h in enumerate((10.0, 30.0, 60.0)):
        pre = r"EEG alone holds " + (r"[0-9.]+ / " * i)
        c.anchored(f"F12 AUC excluding N1, EEG alone, h={h:.0f}s",
                   pre + NUM,
                   at("eeg_only", h)["excluding_n1_transitions"]["roc_auc"])
    # The claim is that the baseline is at chance. Assert it, do not just quote
    # it: 0.5 within 0.02 across every horizon.
    so_ex = [at("stage_only", h)["excluding_n1_transitions"]["roc_auc"]
             for h in (10.0, 30.0, 60.0)]
    c.check("F12 stage-only baseline really is at chance without N1",
            all(abs(v - 0.5) < 0.02 for v in so_ex),
            " ".join(f"{v:.3f}" for v in so_ex))
    c.check("F12 EEG really does beat that baseline without N1",
            all(at("eeg_only", h)["excluding_n1_transitions"]["roc_auc"]
                > at("stage_only", h)["excluding_n1_transitions"]["roc_auc"] + 0.05
                for h in (10.0, 30.0, 60.0)),
            "EEG > stage-only by >0.05 at all three horizons")

    ceiling = st["transition_detection"]["label_noise_ceiling"]
    kappa_n1 = ceiling["per_stage_kappa"]["N1"]
    c.anchored("F12 N1 kappa (results)",
               r"scored stage\s*\n?\(\$\\kappa = " + NUM, kappa_n1)
    c.anchored("F12 N1 kappa (glossary)",
               r"agree on least \(\$\\kappa = " + NUM, kappa_n1)
    c.anchored("F12 overall kappa (results)",
               r"\(\$\\kappa = [0-9.]+\$, against \$" + NUM, ceiling["overall_kappa"])
    c.anchored("F12 overall kappa (glossary)",
               NUM + r" is Sleep-EDF's overall stage-scoring agreement", ceiling["overall_kappa"])
    c.anchored("F12 raw per-epoch scorer agreement",
               r"At roughly " + NUM + r" per-epoch human agreement", ceiling["raw_epoch_agreement"])

    # Balanced accuracy: the headline negative. This is where a number appeared
    # in the primer (0.734) that exists in no artifact and no run log -- see the
    # cycle-5 row in REGISTER.md. Both the results sentence and the summary Q&A
    # restate the pair, in opposite operand order, so anchor each separately.
    def ba(key: str, horizon: float = 60.0) -> float:
        row = next(r for r in st["results"][key] if r["horizon_s"] == horizon)
        return row["all"]["model"]["balanced_accuracy"]

    c.anchored("F12 balanced accuracy, stage only (results)",
               r"balanced accuracy\s*\n?is \\textbf\{" + NUM, ba("stage_only"))
    c.anchored("F12 balanced accuracy, + EEG (results)",
               r"\\textbf\{" + NUM + r"\} with EEG features", ba("eeg_plus_stage"))
    c.anchored("F12 balanced accuracy, + all PSG (results)",
               r"and \\textbf\{" + NUM + r"\} using all polysomnography",
               ba("all_plus_stage"))
    c.anchored("F12 balanced accuracy, EEG alone (results)",
               r"balanced accuracy falls to \\textbf\{" + NUM, ba("eeg_only"))
    c.anchored("F12 balanced accuracy, + EEG (summary)",
               r"no change \(" + NUM, ba("eeg_plus_stage"))
    c.anchored("F12 balanced accuracy, stage only (summary)",
               r"no change \([0-9.]+ versus " + NUM, ba("stage_only"))

    # The stage-only model must reproduce persistence exactly; the primer sells
    # that as a harness self-check, so it has to actually hold.
    so_row = next(r for r in st["results"]["stage_only"] if r["horizon_s"] == 60.0)
    c.check("F12 stage-only model reproduces persistence exactly (harness check)",
            so_row["all"]["model"]["balanced_accuracy"]
            == so_row["all"]["persistence"]["balanced_accuracy"],
            f"{so_row['all']['model']['balanced_accuracy']:.6f}")

    # Grid-edge selections are a known failure mode in this project (it has been
    # caught by them twice). Surface them rather than let them pass silently.
    #
    # `hp_at_grid_edge` is one of "no" / "tied" / "edge" (older artifacts store a
    # bool). Only "edge" is a bug: "tied" means every grid value scored
    # identically, so the tie-break picked index 0 and there is nothing outside
    # the grid to find. Treating "tied" as an edge would flag the stage-only
    # model forever, since its one-hot features make regularisation irrelevant
    # by construction. Note `"no"` is truthy, so this must compare, not coerce.
    def _is_edge(v) -> bool:
        return v is True if isinstance(v, bool) else v == "edge"

    edges = [(k, r["horizon_s"], r["selected_hp"])
             for k, rows in list(st["results"].items())
             + list(st["transition_detection"]["by_input"].items())
             for r in rows if _is_edge(r.get("hp_at_grid_edge"))]
    if edges:
        print(f"      NOTE: {len(edges)} grid-edge hyper-parameter selections in "
              f"the F12 artifact: {edges[:4]}{' ...' if len(edges) > 4 else ''}")
    c.check("F12 grid-edge selections are disclosed in the primer",
            not edges or "grid edge" in c.tex.lower(),
            f"{len(edges)} edge hits")

    # ---- Sample sizes must be stated, and must match the artifacts ----------
    # F12 was quoted beside F11 without saying it ran on 10 subjects to F11's
    # 60, which reads as though they were comparably powered. A number's sample
    # size is part of the number.
    c.anchored("F12 subject count stated in primer matches artifact",
               r"re-run on \\textbf\{" + NUM + r" subjects\}",
               float(st["n_subjects"]), tol=0.5)
    c.anchored("F12 test-subject count stated in primer matches artifact",
               r"select / \\textbf\{" + NUM + r" test\}", float(len(st["test"])), tol=0.5)
    # While F12 was under-powered the primer carried a "Provisional" box. That
    # box is gone now that the run is at full size, so this check would be
    # vacuously true forever. It is replaced by the obligation that outlives it:
    # the 10-subject conclusion was *published* and then reversed, so the
    # retraction has to stay visible where a reader of the old number will meet
    # it. Both halves are required -- the sample size that caused it, and the
    # direction it moved.
    if st["n_subjects"] < 60:
        c.check("F12 provisional caveat present while n_subjects < 60",
                "Provisional" in c.tex, f"n_subjects={st['n_subjects']}")
    else:
        c.check("F12 retraction of the 10-subject caveat stays visible",
                "10 subjects" in c.tex.replace("\\textbf{", "").replace("}", "")
                and "reversed it: excluding N1 is where the effect is" in c.tex,
                "retraction narrative present at full sample size")

    n_f11 = {h["eeg"]["paired_vs_persistence"]["n"]
             for bd in env["bands"].values() for h in bd["horizons"]}
    c.check("F11 test-subject count consistent across all cells", len(n_f11) == 1,
            f"{sorted(n_f11)}")

    # ---- F10: the falsified correlation extension ---------------------------
    corr = CORRELATION.read_text()
    # The results doc lists several "N / 286" rows -- the bound itself holds
    # 0/286, the W/H shortcut fails 223/286. Anchor on the shortcut row by name
    # so this cannot silently latch onto the passing row above it.
    m = re.search(r"`W/H` shortcut carried over from F9 \| \*\*(\d+)\s*/\s*286\*\*", corr)
    c.check("F10 violation count sourced from the W/H row of the results doc",
            m is not None, f"source={m.group(1) + '/286' if m else 'row not found'}")
    if m:
        c.anchored("F10 violation count in primer",
                   r"Violations in " + NUM + r" of 286 cells", float(m.group(1)), tol=0.5)
    c.present("F10 median s_pred = 0.700", "0.700")
    c.present("F10 worst-case factor 10.8x", "10.8")
    c.present("F10 threshold for the historical bug", "0.99213")

    # ---- Integrity: the retraction must stay on the record ------------------
    reg = REGISTER.read_text()
    c.check("REGISTER still records the retracted cross-dataset claim",
            "retracted a claim" in reg and "D3 stays open" in reg)
    c.check("primer devotes a subsection to the retraction",
            "\\subsection{The retraction" in c.tex)
    # The retraction had two independent causes and the subtler one -- that the
    # two numbers were never the same estimand -- is the one that would quietly
    # go missing in an edit, leaving only "the baseline was under-tuned". Check
    # the admission in the narrative, not just the glossary entry.
    # The retraction had two independent causes, and the subtler one -- that the
    # two numbers were never the same estimand -- is the one that would quietly
    # go missing in an edit, leaving only "the baseline was under-tuned". It is
    # stated in three places; require all three, since losing any one weakens
    # the admission while leaving the others to pass a laxer check.
    c.check("estimand: glossary entry", "Estimand &" in c.tex)
    c.check("estimand: transferable-lesson keybox",
            "is it the same estimand" in c.tex)
    c.check("estimand: retraction narrative",
            "not the same estimand" in c.tex or "never the same estimand" in c.tex)

    # ---- Structural ---------------------------------------------------------
    parts = re.findall(r"\\section\{Part ([IVX]+) ---", c.tex)
    c.check("primer parts are numbered I-X with no gaps or repeats",
            parts == ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"],
            f"found {parts}")
    return c


def self_test() -> int:
    """Plant known faults in a copy of the primer; every one must be caught.

    A verifier that has never been observed to fail is not evidence of anything.
    """
    orig = TEX.read_text()
    faults = [
        ("stale F11 correlation (result site)", "correlation is $r = 0.294", "correlation is $r = 0.394"),
        ("stale F11 correlation (control site)", "falls from $r = 0.294", "falls from $r = 0.291"),
        ("stale residualised r", "to $r = 0.181$", "to $r = 0.281$"),
        ("stale F12 AUC (results)", "\\textbf{0.669} (with EEG)", "\\textbf{0.769} (with EEG)"),
        ("F12 results/summary disagree", "AUC 0.625\n$\\rightarrow$ 0.669",
         "AUC 0.625\n$\\rightarrow$ 0.665"),
        ("N1 share altered", "\\textbf{58\\,\\%} of transitions involve stage N1",
         "\\textbf{52\\,\\%} of transitions involve stage N1"),
        ("kappa altered (results)", "scored stage\n($\\kappa = 0.24$", "scored stage\n($\\kappa = 0.94$"),
        ("kappa altered (glossary only)", "agree on least ($\\kappa = 0.24$)",
         "agree on least ($\\kappa = 0.44$)"),
        ("base rate altered", "base rate of \\textbf{0.155}", "base rate of \\textbf{0.095}"),
        # The reversal itself: the N1-excluded numbers are the whole new claim,
        # so each of the three quoted horizons must be individually load-bearing.
        ("excluding-N1 baseline AUC altered (h=10)",
         "chance} --- AUC \\textbf{0.501", "chance} --- AUC \\textbf{0.601"),
        ("excluding-N1 baseline AUC altered (h=60)",
         "0.511 / 0.486}", "0.511 / 0.586}"),
        ("excluding-N1 EEG AUC altered (h=10)",
         "EEG alone holds 0.691", "EEG alone holds 0.791"),
        ("excluding-N1 EEG AUC altered (h=60)",
         "0.691 / 0.691 / 0.599", "0.691 / 0.691 / 0.699"),
        ("F12 all-PSG balanced accuracy altered",
         "and \\textbf{0.821} using all polysomnography",
         "and \\textbf{0.921} using all polysomnography"),
        ("retraction subsection removed", "\\subsection{The retraction", "\\subsection{The omission"),
        ("estimand admission removed", "same estimand", "same thing"),
        ("part renumbered", "\\section{Part IX --- Glossary}", "\\section{Part XI --- Glossary}"),
        ("F10 count altered", "Violations in 223 of 286", "Violations in 23 of 286"),
        ("F12 subject count misstated", "re-run on \\textbf{60 subjects}",
         "re-run on \\textbf{10 subjects}"),
        ("F12 test-subject count misstated", "\\textbf{15\ntest}", "\\textbf{3\ntest}"),
        # The retraction is the claim most convenient to quietly delete, so it
        # gets its own fault: overwrite the reversal rather than disclosing it.
        ("F12 retraction quietly overwritten", "reversed it: excluding N1",
         "confirmed it: excluding N1"),
    ]
    missed, vacuous = [], []
    try:
        for name, old, new in faults:
            if old not in orig:
                vacuous.append(name)
                print(f"VACUOUS {name}  [pattern absent: {old!r}]")
                continue
            TEX.write_text(orig.replace(old, new, 1))
            caught = build(TEX.read_text()).failures
            print(f"{'CAUGHT ' if caught else 'MISSED '} {name}")
            if not caught:
                missed.append(name)
    finally:
        TEX.write_text(orig)

    clean = not build(TEX.read_text()).failures
    print(f"\nrestored file verifies clean: {'yes' if clean else 'NO'}")
    ok = not missed and not vacuous and clean
    print(f"SELF-TEST {'PASS' if ok else 'FAIL'}: "
          f"{len(faults) - len(missed) - len(vacuous)}/{len(faults)} faults caught"
          + (f", missed={missed}" if missed else "")
          + (f", vacuous={vacuous}" if vacuous else ""))
    return 0 if ok else 1


def main() -> int:
    if not TEX.exists():
        print(f"missing {TEX}", file=sys.stderr)
        return 2
    if "--self-test" in sys.argv:
        return self_test()
    c = build(TEX.read_text())
    print(f"\n{c.n - len(c.failures)}/{c.n} checks passed")
    if c.failures:
        print("FAILED: " + ", ".join(c.failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
