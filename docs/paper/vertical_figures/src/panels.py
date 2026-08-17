"""Detect panel bounding boxes in the wide multi-panel source figures.

Pure geometry. We recursively split a region wherever a *low ink density*
band runs the full width or height of that region, alternating axes. Strict
zero-ink splitting is not enough because adjacent matplotlib panels often
have a stray tick label or legend that bridges the visual gutter.

Nothing here draws or synthesizes content. The boxes produced are the only
regions the composer is permitted to copy, which is what keeps the output
provably a crop of the original artwork.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source_wide"
BG_TOL = 12  # per-channel distance from white still counted as background


def ink_mask(im: Image.Image) -> np.ndarray:
    """True where the pixel is *not* background."""
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    return (255 - a).max(axis=2) > BG_TOL


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs where flags is True."""
    out: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def _gap_runs(mask: np.ndarray, axis: int, density: float, min_gap: int) -> list[tuple[int, int]]:
    """Interior low-density bands along `axis` that are at least min_gap wide."""
    prof = mask.sum(axis=1 - axis)
    span = mask.shape[1 - axis]
    quiet = prof <= max(1, int(density * span))
    runs = [r for r in _runs(quiet) if r[1] - r[0] >= min_gap]
    n = mask.shape[axis]
    return [r for r in runs if r[0] > 0 and r[1] < n]  # interior only


def _tight(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Shrink a box to its ink."""
    y0, y1, x0, x1 = box
    sub = mask[y0:y1, x0:x1]
    ys = np.where(sub.any(axis=1))[0]
    xs = np.where(sub.any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    return (y0 + int(ys[0]), y0 + int(ys[-1]) + 1, x0 + int(xs[0]), x0 + int(xs[-1]) + 1)


def _split_once(mask: np.ndarray, box: tuple[int, int, int, int], axis: int,
                density: float, min_gap: int) -> list[tuple[int, int, int, int]] | None:
    """Split box along axis at its widest interior quiet band. None if no split."""
    y0, y1, x0, x1 = box
    sub = mask[y0:y1, x0:x1]
    gaps = _gap_runs(sub, axis, density, min_gap)
    if not gaps:
        return None
    base = y0 if axis == 0 else x0
    cuts = [base] + [base + (g[0] + g[1]) // 2 for g in gaps] + [base + sub.shape[axis]]
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        piece = (a, b, x0, x1) if axis == 0 else (y0, y1, a, b)
        t = _tight(mask, piece)
        if t is not None:
            parts.append(t)
    return parts if len(parts) > 1 else None


def _grid(mask: np.ndarray, root: tuple[int, int, int, int], density: float,
          min_gap_frac: float) -> list[tuple[int, int, int, int]]:
    """Split into a plot grid: rows first, then columns within each row.

    Depth is capped at two on purpose. These sources are matplotlib subplot
    grids, so one row pass plus one column pass recovers exactly the panels.
    Recursing deeper starts carving up the *insides* of a single diagram.
    """
    h, w = mask.shape
    row_gap = max(8, int(min_gap_frac * h))
    col_gap = max(8, int(min_gap_frac * w))
    rows = _split_once(mask, root, 0, density, row_gap) or [root]
    out: list[tuple[int, int, int, int]] = []
    for r in rows:
        out.extend(_split_once(mask, r, 1, density, col_gap) or [r])
    return out


def _absorb_fragments(boxes: list[tuple[int, int, int, int]],
                      frac_of_max: float) -> list[tuple[int, int, int, int]]:
    """Merge slivers (axis labels, caption strips, tick rows) into real panels.

    A fragment is identified per *dimension*, not by area. A caption strip can
    be very wide, so its area clears any sensible area threshold while its
    height is a fifth of a real panel's. A genuine panel is substantial in both
    width and height relative to the largest box; a sliver is thin in at least
    one. The largest box anchors the comparison because it is a stable proxy
    for "one real panel", whereas a median collapses toward the fragments once
    several appear.

    Fragments are merged into the nearest surviving panel rather than dropped,
    since a shared y-axis label or a footnote is genuine content from the
    original figure and discarding it would silently alter the artwork.
    """
    if len(boxes) < 2:
        return boxes
    cur = list(boxes)
    for _ in range(8):  # converges quickly; bounded so it cannot spin
        max_w = max(b[3] - b[2] for b in cur)
        max_h = max(b[1] - b[0] for b in cur)
        is_panel = lambda b: (  # noqa: E731
            (b[3] - b[2]) >= frac_of_max * max_w and (b[1] - b[0]) >= frac_of_max * max_h)
        big = [b for b in cur if is_panel(b)]
        small = [b for b in cur if not is_panel(b)]
        if not small or not big:
            return cur
        for s in small:
            cy, cx = (s[0] + s[1]) / 2, (s[2] + s[3]) / 2
            j = min(range(len(big)), key=lambda i: (
                max(0, big[i][0] - cy, cy - big[i][1]) ** 2
                + max(0, big[i][2] - cx, cx - big[i][3]) ** 2))
            y0, y1, x0, x1 = big[j]
            big[j] = (min(y0, s[0]), max(y1, s[1]), min(x0, s[2]), max(x1, s[3]))
        cur = big
    return cur


def detect(path: Path, density: float = 0.004, min_gap_frac: float = 0.018,
           frac_of_max: float = 0.30) -> list[dict]:
    """Return panel boxes as dicts with x, y, w, h in native source pixels."""
    im = Image.open(path)
    mask = ink_mask(im)
    h, w = mask.shape
    root = _tight(mask, (0, h, 0, w))
    if root is None:
        return []
    raw = _grid(mask, root, density, min_gap_frac)
    raw = _absorb_fragments(raw, frac_of_max)
    boxes = [{"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0} for y0, y1, x0, x1 in raw]
    boxes.sort(key=lambda b: (b["y"] // max(1, h // 12), b["x"]))
    return boxes


def main() -> None:
    report = {}
    for p in sorted(SRC.glob("*.png")):
        im = Image.open(p)
        boxes = detect(p)
        report[p.name] = {"size": list(im.size), "n_panels": len(boxes), "panels": boxes}
        print(f"{p.name}  {im.size[0]}x{im.size[1]}  panels={len(boxes)}")
        for i, b in enumerate(boxes):
            print(f"    [{i}] x={b['x']:5d} y={b['y']:5d} w={b['w']:5d} h={b['h']:5d}")
    (ROOT / "panels.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
