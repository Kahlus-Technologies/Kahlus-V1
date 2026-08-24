# Generated art drop zone

FigureLabs exports go here, named exactly as below. `compose_figures.py` reads
these paths; anything else is ignored.

    head_montage.png      prompt 1   Fig 1 panel A
    latent_field.png      prompt 2   Fig 1 panel B
    architecture.png      prompt 3   Figure 2 (whole figure)
    overlap.png           prompt 4   Fig 3 panel A
    subject_split.png     prompt 5   Fig 3 panel B

Prompts are in ../FIGURELABS_PROMPTS.md.

Two requirements, both enforced at build time:

- **>= 1600 px on the long edge.** Below that the panel cannot print at 600 dpi
  in an IEEE column, which is what produced the blur complaint on Version 6.
- **No metrics in the art.** Numbers are overlaid from CSV so they cannot drift
  from the artifact. The old Figure 1 has MSE=3.116 / r=0.972 baked in, and
  Section VI retracts exactly those values.
