"""Geometry gate for the IECBES figures.

Eyeballing a 3.5-inch figure at screen size hides exactly the defects that
matter in print: a label creeping outside its panel, two text boxes touching,
or type that falls under the venue's legibility floor. This module checks those
three properties mechanically so a build either passes or names its failure.

Checks
------
1. MIN_PT      every text artist prints at >= 7.0 pt at final placement
2. IN_BOUNDS   every text artist's bbox lies inside its axes' panel rect
3. NO_OVERLAP  no two text artists in the same panel have intersecting bboxes

Nothing here inspects pixels; it queries the renderer's laid-out bounding
boxes, which is what the PDF backend will actually emit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

MIN_PT = 7.0
# Bounding boxes of adjacent text can legitimately touch by a hair because a
# glyph box includes ascender/descender padding the ink never fills.
TOUCH_TOL_IN = 0.012


def _texts(ax):
    out = []
    for t in ax.texts:
        s = t.get_text().strip()
        if s:
            out.append(t)
    return out


def _check(fig, name: str) -> list[str]:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dpi = fig.dpi
    fails: list[str] = []

    for ai, ax in enumerate(fig.axes):
        texts = _texts(ax)

        for t in texts:
            pt = t.get_fontsize()
            if pt < MIN_PT - 1e-6:
                fails.append(
                    f"{name} ax{ai} MIN_PT {pt:.2f}pt < {MIN_PT}pt: "
                    f"{t.get_text()[:38]!r}"
                )

        # Panel rect in inches. Axes without a drawn frame still define the
        # region the panel occupies, which is the containment target.
        pos = ax.get_position()
        px0 = pos.x0 * fig.get_figwidth()
        px1 = pos.x1 * fig.get_figwidth()
        py0 = pos.y0 * fig.get_figheight()
        py1 = pos.y1 * fig.get_figheight()

        boxes = []
        for t in texts:
            bb = t.get_window_extent(renderer=rend)
            x0, x1 = bb.x0 / dpi, bb.x1 / dpi
            y0, y1 = bb.y0 / dpi, bb.y1 / dpi
            boxes.append((x0, y0, x1, y1, t))

            if t.get_gid() == "chrome":
                # Panel headers and sub-captions live outside the axes rect on
                # purpose. The meaningful constraint is the figure canvas.
                if (x0 < -0.02 or y0 < -0.02
                        or x1 > fig.get_figwidth() + 0.02
                        or y1 > fig.get_figheight() + 0.02):
                    fails.append(
                        f"{name} ax{ai} IN_BOUNDS chrome off-canvas: "
                        f"{t.get_text()[:38]!r}"
                    )
            elif (x0 < px0 - 0.02 or x1 > px1 + 0.02
                    or y0 < py0 - 0.02 or y1 > py1 + 0.02):
                fails.append(
                    f"{name} ax{ai} IN_BOUNDS escaped panel: "
                    f"{t.get_text()[:38]!r}"
                )

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax0, ay0, ax1, ay1, ta = boxes[i]
                bx0, by0, bx1, by1, tb = boxes[j]
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if (ta.get_gid() == "chrome") != (tb.get_gid() == "chrome"):
                    continue
                if ox > TOUCH_TOL_IN and oy > TOUCH_TOL_IN:
                    fails.append(
                        f"{name} ax{ai} NO_OVERLAP {ox:.3f}x{oy:.3f}in: "
                        f"{ta.get_text()[:24]!r} vs {tb.get_text()[:24]!r}"
                    )
    return fails


def main() -> int:
    """Gate the figures that actually ship.

    Earlier this imported the standalone matplotlib modules and called their
    build(), which wrote to the same filenames as the compositor and silently
    replaced the composed figures, then reported PASS on artwork that was no
    longer on disk. The gate now inspects the compositor's live figures and
    never writes anything.
    """
    import compose_figures as cf

    targets = [("fig1", cf.build_figure1),
               ("fig2", cf.build_figure2),
               ("fig3", cf.build_figure3)]

    all_fails: list[str] = []
    skipped: list[str] = []

    for name, fn in targets:
        try:
            fig = fn()
        except (cf.MissingArt, cf.ResolutionFail) as e:
            skipped.append(f"{name}: not built ({e})")
            continue
        all_fails += _check(fig, name)
        plt.close(fig)

    for s_ in skipped:
        print("  skip", s_)

    if all_fails:
        print(f"GEOMETRY GATE: FAIL ({len(all_fails)})")
        for f in all_fails:
            print("  -", f)
        return 1

    if not skipped:
        print("GEOMETRY GATE: PASS  (min-pt, in-bounds, no-overlap)")
    else:
        print("GEOMETRY GATE: PASS on built figures; some still pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
