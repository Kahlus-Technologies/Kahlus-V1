"""Build the final paper figures as hybrid vertical columns.

FIGURE PLAN
-----------
Each figure opens with figurelabs artwork that establishes *what is being done*,
then continues with matplotlib panels that establish *what was measured*. The
art earns its place by replacing weaker schematic panels, not by decorating.

fig1_acquisition  (a) figurelabs scalp-cap render  <- replaces the plain
                      circle-with-dots schematic that image1(a) used
                  (b) nuisance decomposition, image1(b)
                  (c) leakage inflation bars,   image1(c)
                  (d) honest error vs horizon,  image1(d)

fig2_protocol     (a-d) the two-protocol argument, image2. Already strong and
                      self-contained; no art needed, and adding some would
                      dilute a clean evidence chain.

fig3_latent       (a) figurelabs latent-field strip, the paper's core object
                  (b-d) image4 gate/RFS panels

fig4_gate         image3 as-is: a compact 3-panel gate result.
fig5_sweep        image5 as-is: horizon sweep.

Panels dropped on purpose are recorded in `plan.json` with a reason, so the
substitution is auditable rather than silent.
"""

from __future__ import annotations

import json
from pathlib import Path

from hybrid import compose

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures_final"

PANELS = json.loads((ROOT / "panels.json").read_text())


def p(src: str, i: int) -> dict:
    return {"kind": "data", "source": src, "crop": PANELS[src]["panels"][i]}


def art(key: str) -> dict:
    return {"kind": "art", "asset": key}


PLAN = [
    ("fig1_acquisition.png", "image1.png",
     [art("eeg_windowing"), p("image1.png", 1), p("image1.png", 2), p("image1.png", 3)],
     "Fig. 1. Noninvasive acquisition and what must be removed before a claim.",
     {"dropped": ["image1(a)"],
      "reason": "circle-with-dots schematic replaced by the figurelabs EEG windowing "
                "band, which shows the cap and the input window in one image"}),

    ("fig2_protocol.png", "image2.png",
     [p("image2.png", i) for i in range(4)],
     "Fig. 2. Two evaluation protocols and the scale of the leakage illusion.",
     {"dropped": [], "reason": "self-contained evidence chain"}),

    ("fig3_latent.png", "image4.png",
     [art("latent_field")] + [p("image4.png", i) for i in range(1, 4)],
     "Fig. 3. Latent field and gate outcomes under held-out evaluation.",
     {"dropped": ["image4(a)"],
      "reason": "opening schematic replaced by the figurelabs latent-field strip"}),

    ("fig4_gate.png", "image3.png",
     [p("image3.png", i) for i in range(3)],
     "Fig. 4. Gate protocol results.",
     {"dropped": [], "reason": "compact 3-panel result"}),

    ("fig5_sweep.png", "image5.png",
     [p("image5.png", i) for i in range(3)],
     "Fig. 5. Forecast horizon sweep.",
     {"dropped": [], "reason": "compact 3-panel result"}),
]


def main() -> None:
    prov, plan_log = [], []
    for fname, src, rows, cap, note in PLAN:
        rec = compose(src, rows, OUT / fname, cap)
        prov.append(rec)
        n_art = sum(1 for r in rows if r["kind"] == "art")
        plan_log.append({"figure": fname, "n_art": n_art,
                         "n_data": len(rows) - n_art, **note})
        print(f"{fname:22s} {rec['canvas']['w']}x{rec['canvas']['h']} "
              f"aspect {rec['aspect']:.2f}  {rec['effective_dpi']:.0f}dpi "
              f"@{rec['placed_width_in']:.2f}in  art={n_art} data={len(rows)-n_art}")
    (ROOT / "provenance_final.json").write_text(json.dumps(prov, indent=2))
    (ROOT / "plan.json").write_text(json.dumps(plan_log, indent=2))


if __name__ == "__main__":
    main()
