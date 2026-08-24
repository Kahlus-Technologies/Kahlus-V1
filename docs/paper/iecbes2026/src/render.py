"""SVG rendering, with the library path macOS needs.

cairosvg on this machine reports a missing ``libcairo-2.dll``, which is a
misleading way of saying it could not find libcairo at all. Homebrew has it at
/opt/homebrew/lib, so the fix is to make that visible before the import.

Kept in one place because every verification path renders, and a verification
step that cannot run is worse than no verification step: it gets skipped, and
then the build reports success on artwork nobody looked at.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BREW = "/opt/homebrew/lib"
if Path(_BREW).is_dir():
    cur = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if _BREW not in cur:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{_BREW}:{cur}".rstrip(":")

try:
    import cairosvg
except OSError as e:  # pragma: no cover - environment specific
    raise SystemExit(
        f"cairosvg could not load libcairo ({e}).\n"
        f"On macOS: brew install cairo, then re-run."
    )


def svg_to_png(svg: Path, png: Path, dpi: float = 600.0) -> Path:
    png.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(svg), write_to=str(png), dpi=dpi,
                     background_color="white")
    return png


def svg_to_png_transparent(svg: Path, png: Path, dpi: float = 150.0) -> Path:
    """Render without a white backdrop, so ink can be found via alpha."""
    png.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(svg), write_to=str(png), dpi=dpi)
    return png


def svg_to_pdf(svg: Path, pdf: Path) -> Path:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2pdf(url=str(svg), write_to=str(pdf))
    return pdf


if __name__ == "__main__":
    src = Path(sys.argv[1])
    svg_to_png(src, src.with_suffix(".png"))
    print("rendered", src.with_suffix(".png"))
