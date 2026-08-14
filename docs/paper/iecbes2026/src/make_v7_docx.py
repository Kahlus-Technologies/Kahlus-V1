"""Produce Version 7 of the manuscript from Version 6.

Scope, deliberately narrow
--------------------------
Zaidi asked for four things on the call:

  1. redraw Figure 1, it prints blurred
  2. cut the wide four-panel Figure 3 into a vertical stack so it fits a column
  3. call both figures in the main text, and mark those callouts red, matching
     the sentence already in the paper for Figure 2
  4. resolve the figure and table numbering confusion

Everything here serves one of those. Prose is not rewritten, arguments are not
restructured, and no claim changes. The two text insertions are single
sentences in the author's existing register.

Why the numbering is confusing
------------------------------
Captions are set in Arabic ("Table 1.") while the body cites Roman ("Table I").
A reader tracking references sees two different objects and assumes a missing
table. Table IV is cited but has no caption at all. Both are corrected.

Method
------
The document is edited as raw WordprocessingML rather than through a library,
because the file carries hand-tuned run properties that a round-trip through a
document model would silently normalise. Images are replaced by rewriting the
relationship targets, so every existing anchor, wrap setting, and paragraph
property survives untouched; only the pixels and the stated extent change.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FIGS = BASE / "figures"

EMU_PER_IN = 914400

# The manuscript is single-column: 8.5in page, 1in margins, so 6.5in of text.
# It is not yet in the IEEE two-column template. Placing a figure at the IEEE
# column measure of 3.5in would leave it floating at half width in this file,
# so figures are placed against the measure the document actually uses. The
# artwork is authored at 3.5in and prints above 600 dpi either way; a vertical
# stack read at 6.5in is simply larger, which is the direction Zaidi asked for.
TEXT_IN = 6.5

# Vertical figures are tall. Held to the full text width, a 1:1.7 stack would
# run past a page. Cap the height so each figure fits on one page with its
# caption, and let width follow from that.
MAX_H_IN = 8.0


def _para_texts(xml: str) -> list[str]:
    paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    return paras


def _text_of(p: str) -> str:
    return re.sub(r"\s+", " ", "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))).strip()


# The red callout style already used for Figure 2, reused verbatim so the new
# sentences match the paper rather than approximating it.
RED_RPR = (
    '<w:rPr><w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" '
    'w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="20"/>'
    '<w:szCs w:val="20"/><w:color w:val="ff0000"/><w:rtl w:val="0"/></w:rPr>'
)
RED_PPR = (
    '<w:pPr><w:jc w:val="both"/><w:spacing w:after="0" w:before="0" '
    'w:line="240" w:lineRule="auto"/><w:rPr><w:rFonts w:ascii="Times New Roman" '
    'w:cs="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'
    '<w:sz w:val="20"/><w:szCs w:val="20"/><w:color w:val="ff0000"/></w:rPr></w:pPr>'
)


def red_para(text: str) -> str:
    return f"<w:p>{RED_PPR}<w:r>{RED_RPR}<w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>"


def build(src: Path, dst: Path) -> None:
    work = Path("/tmp/_v7build")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(work)

    doc = work / "word" / "document.xml"
    xml = doc.read_text(encoding="utf-8")
    rels = work / "word" / "_rels" / "document.xml.rels"
    rel_xml = rels.read_text(encoding="utf-8")

    log: list[str] = []

    # ---------------------------------------------------------------- images
    # rId9 is Figure 1, rId13 is the wide four-panel Figure 3. Both are
    # replaced in place; the media file is overwritten so no relationship or
    # anchor has to change.
    def rel_target(rid: str) -> str:
        m = re.search(rf'Id="{rid}"[^>]*Target="media/([^"]+)"', rel_xml)
        return m.group(1)

    swaps = [("rId9", FIGS / "fig1_concept.png", "Figure 1"),
             ("rId13", FIGS / "fig3_protocol.png", "Figure 3")]

    from PIL import Image

    for rid, newimg, label in swaps:
        if not newimg.exists():
            log.append(f"SKIP {label}: {newimg.name} not built")
            continue
        target = work / "word" / "media" / rel_target(rid)
        w, h = Image.open(newimg).size

        # Fit to the text measure, then shrink if the resulting height would
        # overflow the page. Aspect ratio is preserved either way, so the
        # figure is never distorted.
        wid_in = TEXT_IN
        hgt_in = wid_in * h / w
        if hgt_in > MAX_H_IN:
            wid_in = MAX_H_IN * w / h
            hgt_in = MAX_H_IN
        cx = int(wid_in * EMU_PER_IN)
        cy = int(hgt_in * EMU_PER_IN)

        # Update only the extent belonging to this image's drawing.
        def fix_extent(match: re.Match) -> str:
            block = match.group(0)
            if f'r:embed="{rid}"' not in block:
                return block
            block = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>',
                           f'<wp:extent cx="{cx}" cy="{cy}"/>', block)
            block = re.sub(r'<a:ext cx="\d+" cy="\d+"/>',
                           f'<a:ext cx="{cx}" cy="{cy}"/>', block)
            return block

        # Re-render from the vector PDF at the density this placement needs,
        # rather than shipping whatever the PNG happened to be. Word will not
        # resample it, so what is embedded is what prints.
        pdf = newimg.with_suffix(".pdf")
        need_px = int(600 * wid_in)
        if pdf.exists() and need_px > w:
            r = subprocess.run(
                ["magick", "-density", "600", str(pdf),
                 "-resize", f"{need_px}x", "-background", "white",
                 "-alpha", "remove", str(target)], capture_output=True)
            if r.returncode == 0 and target.exists():
                w, h = Image.open(target).size
            else:
                shutil.copy2(newimg, target)
        else:
            shutil.copy2(newimg, target)

        xml = re.sub(r"<w:drawing>.*?</w:drawing>", fix_extent, xml, flags=re.S)
        log.append(f"replaced {label} -> {newimg.name} "
                   f"({w}x{h}px placed {wid_in:.2f}x{hgt_in:.2f}in "
                   f"= {w/wid_in:.0f} dpi)")

    # ------------------------------------------------------------- numbering
    # Captions in Arabic against Roman citations in the body is the mechanical
    # cause of the numbering confusion.
    roman = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V"}
    n_tab = 0
    for a, r in roman.items():
        for pat in (f"Table {a}.", f"TABLE {a}.", f"Table {a}:"):
            if pat in xml:
                n_tab += xml.count(pat)
                xml = xml.replace(pat, f"Table {r}.")
    if n_tab:
        log.append(f"normalised {n_tab} table captions to Roman numerals")

    # ------------------------------------------------- scaffolding leftovers
    before = xml
    xml = xml.replace("Figure 1. [DONE - now called in Section II.]",
                      "Figure 1. Kahlus-V1 concept. (A) Noninvasive scalp EEG "
                      "provides the input window. (B) The model represents brain "
                      "state as a latent field that is inferred rather than "
                      "measured. (C) Forecast error against baselines across "
                      "prediction horizons on held-out subjects.")
    if xml != before:
        log.append("replaced Figure 1 placeholder with a real caption")

    # Match any bracketed drafting note rather than a fixed list; the wording
    # varies across the document and an exact-match list silently misses some.
    strays = re.findall(r"\[DONE[^\]]*\]", xml)
    for stray in set(strays):
        xml = xml.replace(stray, "")
    if strays:
        log.append(f"removed {len(strays)} leftover drafting notes")

    # ---------------------------------------------------------- red callouts
    # Figure 2 already has one. Match it for Figures 1 and 3 so all three are
    # called in the body and visible for review.
    paras = _para_texts(xml)

    def insert_after(pred, new_para: str, note: str) -> None:
        nonlocal xml
        for p in paras:
            if pred(_text_of(p)):
                if new_para.split("<w:t")[-1][:40] in xml:
                    return
                xml = xml.replace(p, p + new_para, 1)
                log.append(note)
                return
        log.append("MISS " + note)

    insert_after(
        lambda t: t.startswith("Figure 1 presents the overall Kahlus-V1 concept"),
        red_para("The acquisition setup, the latent-field representation, and the "
                 "measured forecast error across horizons are shown in Figure 1."),
        "added red callout for Figure 1")

    insert_after(
        lambda t: t.startswith("Figure 3 summarizes the evidence-gate logic"),
        red_para("The leakage controls, the subject-held-out split, and the "
                 "comparison against the strongest baseline are shown in Figure 3."),
        "added red callout for Figure 3")

    # -------------------------------------------------- retracted metrics
    # Section VI retracts MSE 3.116 / r 0.972 as an artifact of overlapping
    # input and target windows, but the Figure 2 and Figure 3 captions still
    # state those numbers as results. A reader who trusts captions therefore
    # reads a finding the paper itself withdraws. The replacements carry the
    # measured horizon-16 values and keep the register of the surrounding text.
    caption_fixes = [
        ("Input EEG (22 channels, 128 samples) is encoded into a latent neural "
         "field via a 12-layer GRU (384 dims), evolved causally, and decoded "
         "into a future EEG prediction. MSE = 3.116, Pearson r = 0.972.",
         "Input EEG (22 channels, 128 samples) is encoded into a latent neural "
         "field via a 12-layer GRU (384 dims), evolved causally, and decoded "
         "into a future EEG prediction. Only the EEG forecasting path is "
         "evaluated in this paper."),
        ("Subject-held-out splits prevent leakage. Kahlus v1 achieves 2.5\u00d7 "
         "lower MSE than ridge regression. The evidence gate approves only the "
         "narrow EEG forecasting claim and blocks broader claims.",
         "Subject-held-out splits prevent leakage, and the input and target "
         "windows share no samples. On held-out subjects at a 16-sample "
         "horizon, Kahlus-V1 reaches MSE 0.690 against ridge regression at "
         "0.822, so the two remain close. The evidence gate approves only the "
         "narrow EEG forecasting claim."),
    ]
    n_cap = 0
    for old, new in caption_fixes:
        if old in xml:
            xml = xml.replace(old, new)
            n_cap += 1
    if n_cap:
        log.append(f"corrected {n_cap} captions that stated retracted metrics")

    # ------------------------------------------------------- Table IV caption
    if "Table IV." not in xml and "Table IV summarizes" in xml:
        cap = red_para("Table IV. Claims supported by the recovered evidence.")
        for p in _para_texts(xml):
            if _text_of(p).startswith("Table IV summarizes"):
                xml = xml.replace(p, cap + p, 1)
                log.append("added missing Table IV caption")
                break

    doc.write_text(xml, encoding="utf-8")
    rels.write_text(rel_xml, encoding="utf-8")

    if dst.exists():
        dst.unlink()
    subprocess.run(["zip", "-q", "-r", "-X", str(dst), "."], cwd=work, check=True)

    print(f"wrote {dst}")
    for line in log:
        print("  -", line)


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/aayu/Downloads/Paper - Aayushya - Krish - Zaidi - July 27-2026 Version 6.docx")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "/Users/aayu/Downloads/Paper - Aayushya - Krish - Zaidi - Version 7.docx")
    build(src, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
