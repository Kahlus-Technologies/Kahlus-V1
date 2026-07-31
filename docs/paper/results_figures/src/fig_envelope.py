"""Figure: band-envelope forecasting on Sleep-EDF, with the EMG control.

The single-sample voltage forecast dies at roughly half a second, which the
crossover figure shows. The obvious objection is that this kills the whole
premise, so the question becomes whether anything about the signal is
predictable further out. The band envelope is, and by a wide margin: every
horizon out to 60 s beats persistence on 15 of 15 held-out subjects.

The objection to *that* is muscle. Low gamma at the scalp is the band most
contaminated by EMG, and a "neural" forecast that is really jaw tone would look
exactly like this. BNCI has no EMG channel, so on BNCI the objection could not
be answered. Sleep-EDF has EMG submental and EOG, so it can be, and the honest
way to plot it is to draw the residualised result on the same axes rather than
in a supplement: panel (a) shows the raw envelope skill, panel (b) shows what
survives after regressing out concurrent and lagged EMG and EOG.

The gap between the two lines is the part a sceptic would attribute to muscle.
The residualised line staying well above zero is the answer.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import common as C

ARTIFACT = "envelope_sleepedf/results.json"
BANDS = [("low_gamma", "low gamma", "-"), ("beta", "beta", "--")]


def build() -> None:
    C.rc()
    d = C.load(ARTIFACT)

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(C.COL_W, 3.75),
        gridspec_kw={"height_ratios": [1.0, 0.85]},
    )

    manifest_bands = {}

    for band, label, ls in BANDS:
        hz = d["bands"][band]["horizons"]
        hs = [h["horizon_s"] for h in hz]
        # Skill against persistence, as a fraction of the persistence error.
        # Plotting raw MSE would hide the effect because the persistence
        # baseline itself changes with horizon.
        skill = [(h["eeg"]["persistence_mse"] - h["eeg"]["mse"])
                 / h["eeg"]["persistence_mse"] for h in hz]
        r_raw = [h["eeg"]["pearsonr"] for h in hz]
        r_res = [h["eeg_artifact_residualised"]["pearsonr"] for h in hz]
        beats = [h["eeg"]["beats_persistence_n"] for h in hz]
        n_test = hz[0]["eeg"]["n_test_subjects"]

        col = C.C_MODEL if band == "low_gamma" else C.C_RIDGE
        ax.plot(hs, skill, color=col, ls=ls, marker="o", ms=2.6, label=label)
        bx.plot(hs, r_raw, color=col, ls=ls, marker="o", ms=2.6,
                label=f"{label}, raw")
        bx.plot(hs, r_res, color=col, ls=ls, marker="^", ms=2.8, alpha=0.6,
                lw=0.9, label=f"{label}, EMG+EOG removed")

        manifest_bands[band] = {
            "horizons_s": hs, "skill_vs_persistence": skill,
            "pearsonr_raw": r_raw, "pearsonr_residualised": r_res,
            "beats_persistence_n": beats, "n_test_subjects": n_test,
        }

    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xticks([1, 3, 10, 30, 60])
        a.set_xticklabels(["1", "3", "10", "30", "60"])
        a.grid(alpha=0.25, lw=0.4)

    ax.set_ylabel("skill vs persistence")
    ax.set_xlabel("forecast horizon (s, log)")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", frameon=False, handlelength=1.8,
              labelspacing=0.25, borderpad=0.1)

    n_test = d["bands"]["beta"]["horizons"][0]["eeg"]["n_test_subjects"]
    unanimous = all(
        h["eeg"]["beats_persistence_n"] == n_test
        for b in d["bands"] for h in d["bands"][b]["horizons"]
    )
    tag = f"{n_test}/{n_test} subjects, every horizon" if unanimous else "per-subject"
    ax.set_title(f"(a) envelope beats persistence to 60 s\n({tag})", loc="left")

    bx.set_ylabel("Pearson r")
    bx.set_xlabel("forecast horizon (s, log)")
    bx.set_ylim(0, None)
    # Four entries will not fit beside the curves at this width, so the legend
    # goes under the axes in two columns.
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2,
              frameon=False, handlelength=1.7, labelspacing=0.2,
              columnspacing=0.9, borderpad=0.1, fontsize=C.PT_ANNOT)
    bx.set_title("(b) survives the EMG + EOG control", loc="left")

    manifest = {
        "artifact": ARTIFACT,
        "dataset": d["dataset"],
        "n_subjects": d["n_subjects"],
        "n_test_subjects": n_test,
        "residualisation": d["residualisation"],
        "bands": manifest_bands,
    }
    p = C.save(fig, "fig_envelope", manifest)
    print(f"wrote {p}")
    for b in manifest_bands:
        m = manifest_bands[b]
        print(f"  {b}: skill {m['skill_vs_persistence'][0]:.3f} -> "
              f"{m['skill_vs_persistence'][-1]:.3f}, "
              f"r_resid {m['pearsonr_residualised'][0]:.3f} -> "
              f"{m['pearsonr_residualised'][-1]:.3f}, "
              f"beats {set(m['beats_persistence_n'])}/{m['n_test_subjects']}")


if __name__ == "__main__":
    build()
