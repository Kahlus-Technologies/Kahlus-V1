"""Produce Version 7.1 of the manuscript from Version 7.

What this fixes
---------------
Four defects, all found by reading the built document rather than trusting the
build log:

  1. Figure 2 is not the composed architecture figure. It is two thumbnails,
     125x92 and 145x96 pixels, plus an empty text box. At any print size those
     are unreadable.
  2. Tables I, II and III are screenshots, 331 to 343 pixels across. They are
     not selectable, not searchable, not editable by a co-author, and they
     print at roughly 50 dpi.
  3. Table IV is cited twice in Section VII and does not exist.
  4. Four trailing chart images, 2048x559, sit after the references with no
     caption and no callout anywhere in the body.

The tables are rebuilt as native ``<w:tbl>`` (see make_tables.py, which checks
its own numbers against the measured CSV). Figure 2 is replaced with the
composed panel. The orphan charts are removed rather than redrawn: nothing in
the text refers to them, and a figure a reader cannot locate a reason for is
noise.

Everything Version 7 already fixed is inherited by building on top of it.
No prose is rewritten and no claim changes.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BASE = HERE.parent
FIGS = BASE / "figures"

import docx_edit as de          # noqa: E402
import make_tables as mt        # noqa: E402

EMU_PER_IN = 914400
TEXT_IN = 6.5

# The text block is 9in tall. A figure sized to fill it leaves no room for its
# own caption, so Word breaks the page between them and the reader meets an
# unlabelled picture. Cap the artwork so figure and caption fit together.
FIG_MAX_H_IN = 6.4

# 11in page less 1in top and bottom margins.
TEXT_H_IN = 9.0

# Reserved for the caption plus the paragraph spacing around it.
CAPTION_RESERVE_IN = 1.4

# Screenshot -> the table that replaces it. Keyed by media file so a shifting
# relationship id cannot silently retarget the wrong picture.
TABLE_IMAGES = {
    "image15.png": "I",
    "image14.png": "II",
    "image13.png": "III",
}

# The trailing strips after the references. Uncaptioned and uncited.
ORPHAN_IMAGES = {"image2.png", "image3.png", "image5.png", "image6.png"}

# The two thumbnails standing in for Figure 2.
FIG2_THUMBS = {"image16.png", "image17.png"}

# A 1x1 transparent pixel Word attaches to every text box as its Fallback
# picture. It is not content, and treating it as content is what let an empty
# outlined rectangle survive a removal pass that reported success. Detected by
# size rather than by name, so a re-export that renumbers the media cannot
# quietly turn the check off.
SPACER_MAX_PX = 2


def spacer_images(media_dir: Path) -> set[str]:
    from PIL import Image

    found = set()
    for path in media_dir.glob("*"):
        try:
            w, h = Image.open(path).size
        except Exception:                       # noqa: BLE001
            continue
        if w <= SPACER_MAX_PX and h <= SPACER_MAX_PX:
            found.add(path.name)
    return found


def drawing_paragraph_spans(xml: str, rels: dict[str, str], media: set[str]):
    """Spans of top-level paragraphs whose drawings reference any of ``media``."""
    hits = []
    for a, b in de.top_level_paragraphs(xml):
        frag = xml[a:b]
        names = {rels.get(r, r) for r in de.embedded_rids(frag)}
        if names & media:
            hits.append((a, b, names))
    return hits


def empty_shape_spans(xml: str, rels: dict[str, str], spacers: set[str]):
    """Paragraphs holding a floating shape that draws a border around nothing.

    The document has a few of these: a white rectangle with a black outline and
    an empty text box, left behind when whatever was inside it was removed. On
    the page they read as a missing figure, which is worse than nothing at all.

    The catch is that a Word text box is not image-free. Each one carries a
    1x1 transparent spacer as its Fallback picture, so "has no ``r:embed``" is
    not the test for emptiness; "embeds nothing but the spacer" is.
    """
    hits = []
    for a, b in de.top_level_paragraphs(xml):
        frag = xml[a:b]
        if "<w:drawing>" not in frag or "<w:txbxContent>" not in frag:
            continue
        names = {rels.get(r, r) for r in de.embedded_rids(frag)}
        if names - spacers:            # holds a real picture, leave it
            continue
        if de.text_of(frag):           # the box says something
            continue
        hits.append((a, b))
    return hits


def with_keep_next(paragraph: str) -> str:
    """Add ``<w:keepNext/>`` to a paragraph's properties.

    ``w:pPr`` must be the first child of ``w:p``, and ``w:keepNext`` must come
    early within it, so this inserts rather than appends. If the paragraph
    already keeps with the next one, it is returned unchanged.
    """
    if "<w:keepNext/>" in paragraph:
        return paragraph

    open_tag = re.match(r"<w:p\b[^>]*?(/?)>", paragraph)
    if not open_tag:
        return paragraph
    if open_tag.group(1):        # <w:p/>: empty, and there is nothing to keep
        return paragraph
    head, rest = paragraph[:open_tag.end()], paragraph[open_tag.end():]

    if rest.lstrip().startswith("<w:pPr>"):
        i = rest.index("<w:pPr>") + len("<w:pPr>")
        return head + rest[:i] + "<w:keepNext/>" + rest[i:]
    return head + "<w:pPr><w:keepNext/></w:pPr>" + rest


def caption_para(text: str, *, italic: bool = False) -> str:
    rpr = ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" '
           'w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'
           + ("<w:i/>" if italic else "")
           + '<w:sz w:val="18"/><w:szCs w:val="18"/><w:rtl w:val="0"/></w:rPr>')
    return ('<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="120" '
            'w:before="120" w:line="240" w:lineRule="auto"/></w:pPr>'
            f'<w:r>{rpr}<w:t xml:space="preserve">{text}</w:t></w:r></w:p>')


def table_caption_para(text: str) -> str:
    """A table caption in the same face and size Version 7 used.

    The original captions were not body paragraphs: each sat in a text box
    inside the screenshot's drawing, so replacing the screenshot with a real
    table deleted the caption with it. They are re-emitted here as ordinary
    paragraphs at the size the drawing used (8 pt), carrying keepNext so the
    caption cannot be stranded at the foot of a page above its table.
    """
    rpr = ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" '
           'w:eastAsia="Times New Roman" w:hAnsi="Times New Roman"/>'
           '<w:sz w:val="16"/><w:szCs w:val="16"/><w:rtl w:val="0"/></w:rPr>')
    return ('<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/>'
            '<w:spacing w:after="60" w:before="120" w:line="240" '
            'w:lineRule="auto"/></w:pPr>'
            f'<w:r>{rpr}<w:t xml:space="preserve">{text}</w:t></w:r></w:p>')


CAPTION_RE = re.compile(r"Table\s+(?:I{1,3}|IV)\.[^<]*")


def caption_in(span_xml: str) -> str | None:
    """Recover the 'Table N. ...' caption carried inside a drawing."""
    for raw in de.text_runs(span_xml):
        t = html.unescape(raw).strip()
        if CAPTION_RE.fullmatch(t):
            return html.escape(t)
    return None


# Added by the Version 7 pass to guarantee every figure was cited. Each sits
# directly beneath a sentence that already cites the same figure.
DUPLICATE_CALLOUTS = (
    "The acquisition setup, the latent-field representation,",
    "The leakage controls, the subject-held-out split,",
)


def collapse_blank_runs(xml: str, keep: int = 1, threshold: int = 3):
    """Shrink runs of >= threshold empty paragraphs down to `keep` of them."""
    spans = de.top_level_paragraphs(xml)
    blank = []
    for a, b in spans:
        seg = xml[a:b]
        load_bearing = ("<w:drawing" in seg or "w:pict" in seg
                        or "<w:br" in seg or "<w:sectPr" in seg
                        or "<w:tbl" in seg)
        blank.append(not de.text_of(seg) and not load_bearing)

    runs, i = [], 0
    while i < len(spans):
        if blank[i]:
            j = i
            while j < len(spans) and blank[j]:
                j += 1
            if j - i >= threshold:
                runs.append((i + keep, j))
            i = j
        else:
            i += 1

    removed = 0
    for lo, hi in reversed(runs):
        a, b = spans[lo][0], spans[hi - 1][1]
        xml = xml[:a] + xml[b:]
        removed += hi - lo
    return xml, removed


def spacer_para() -> str:
    return ('<w:p><w:pPr><w:spacing w:after="0" w:before="0" w:line="240" '
            'w:lineRule="auto"/></w:pPr></w:p>')


_next_pic_id = iter(range(900, 999))


def inline_picture(rid: str, cx: int, cy: int, name: str) -> str:
    """A plain inline picture run: no anchor, no wrap, no floating group.

    The thumbnails being replaced were anchored inside a shape group, which is
    why they could sit at 125 px without anyone noticing a layout problem. An
    inline picture is measured by the paragraph that holds it, so a wrong size
    is visible immediately.
    """
    pid = next(_next_pic_id)          # docPr ids must be unique in the document
    return (
        "<w:r><w:drawing>"
        f'<wp:inline distB="0" distT="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent b="0" l="0" r="0" t="0"/>'
        f'<wp:docPr id="{pid}" name="{name}"/>'
        "<a:graphic><a:graphicData "
        'uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{name}"/>'
        '<pic:cNvPicPr preferRelativeResize="0"/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:srcRect/>'
        "<a:stretch><a:fillRect/></a:stretch></pic:blipFill>"
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/>'
        '</a:xfrm><a:prstGeom prst="rect"/><a:ln/></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline>"
        "</w:drawing></w:r>"
    )


def add_relationship(rel_xml: str, target: str) -> tuple[str, str]:
    used = {int(n) for n in re.findall(r'Id="rId(\d+)"', rel_xml)}
    rid = f"rId{max(used) + 1}"
    entry = (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org'
             f'/officeDocument/2006/relationships/image" Target="media/{target}"/>')
    return rel_xml.replace("</Relationships>", entry + "</Relationships>"), rid


def rasterise(src_png: Path, dst: Path, width_in: float) -> tuple[int, int]:
    """Write ``dst`` at the density this placement needs.

    Word does not resample on the way in, so whatever is embedded is what
    prints. Re-rendering from the vector PDF keeps the placed figure above
    600 dpi instead of inheriting whatever the PNG happened to be.
    """
    from PIL import Image

    pdf = src_png.with_suffix(".pdf")
    need_px = int(600 * width_in)
    if pdf.exists():
        r = subprocess.run(
            ["magick", "-density", "600", str(pdf), "-resize", f"{need_px}x",
             "-background", "white", "-alpha", "remove", str(dst)],
            capture_output=True)
        if r.returncode == 0 and dst.exists():
            return Image.open(dst).size
    shutil.copy2(src_png, dst)
    return Image.open(dst).size


def build(src: Path, dst: Path) -> list[str]:
    work = Path("/tmp/_v71build")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(work)

    doc = work / "word" / "document.xml"
    xml = doc.read_text(encoding="utf-8")
    rels_path = work / "word" / "_rels" / "document.xml.rels"
    rel_xml = rels_path.read_text(encoding="utf-8")
    rels = de.rel_map(rel_xml)
    spacers = spacer_images(work / "word" / "media")

    log: list[str] = []

    # -------------------------------------------------------------- tables
    # Replace from the end of the document backwards so that each edit cannot
    # invalidate the spans of the ones not yet applied.
    mismatches = mt.disjoint_row_check()
    if mismatches:
        raise SystemExit("table values disagree with the CSV: "
                         + "; ".join(mismatches))

    targets = drawing_paragraph_spans(xml, rels, set(TABLE_IMAGES))
    for a, b, names in sorted(targets, reverse=True):
        which = next(TABLE_IMAGES[n] for n in names if n in TABLE_IMAGES)
        rows = mt.TABLES[which][0]
        cap = caption_in(xml[a:b])
        replacement = ((table_caption_para(cap) if cap else "")
                       + mt.build(which) + spacer_para())
        xml = xml[:a] + replacement + xml[b:]
        log.append(f"Table {which}: screenshot -> native table "
                   f"({len(rows) - 1} rows x {len(rows[0])} cols)"
                   + ("" if cap else "  WARNING caption not recovered"))

    # ------------------------------------------------------------- Table IV
    # Cited twice in Section VII, never present. Insert it under its caption.
    span = de.find_paragraph(
        xml, lambda t: t.startswith("Table IV. Claims supported"))
    if span:
        a, b = span
        rows = mt.TABLES["IV"][0]
        xml = xml[:b] + mt.build("IV") + spacer_para() + xml[b:]
        log.append(f"Table IV: created ({len(rows) - 1} rows x {len(rows[0])} cols)")
    else:
        log.append("MISS Table IV caption not found")

    # ------------------------------------------------------------- Figure 2
    # The composed architecture panel replaces two thumbnails and an empty
    # text box. It is placed immediately above its own caption: in Version 7
    # the artwork and the caption landed on different pages, which is the
    # same defect as having no caption.
    fig2 = FIGS / "fig2_architecture.png"
    targets = drawing_paragraph_spans(xml, rels, FIG2_THUMBS)
    if fig2.exists() and targets:
        from PIL import Image

        w, h = Image.open(fig2).size
        wid_in, hgt_in = TEXT_IN, TEXT_IN * h / w
        if hgt_in > FIG_MAX_H_IN:
            wid_in, hgt_in = FIG_MAX_H_IN * w / h, FIG_MAX_H_IN

        media_name = "fig2_architecture.png"
        px = rasterise(fig2, work / "word" / "media" / media_name, wid_in)
        rel_xml, rid = add_relationship(rel_xml, media_name)

        pic = inline_picture(rid, int(wid_in * EMU_PER_IN),
                             int(hgt_in * EMU_PER_IN), media_name)
        para = ('<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="0" '
                'w:before="0" w:line="240" w:lineRule="auto"/></w:pPr>'
                + pic + "</w:p>")

        # Drop every paragraph that held a thumbnail, back to front so the
        # earlier spans stay valid.
        for aa, bb, _ in reversed(targets):
            xml = xml[:aa] + xml[bb:]

        cap = de.find_paragraph(
            xml, lambda t: t.startswith("Figure 2:") or t.startswith("Figure 2."))
        if cap:
            xml = xml[:cap[0]] + para + xml[cap[0]:]
            where = "above its caption"
        else:
            xml = xml[:targets[0][0]] + para + xml[targets[0][0]:]
            where = "at the former thumbnail position"
        log.append(f"Figure 2: thumbnails -> composed panel {where} "
                   f"({px[0]}x{px[1]}px placed {wid_in:.2f}x{hgt_in:.2f}in "
                   f"= {px[0] / wid_in:.0f} dpi)")
    elif not fig2.exists():
        log.append("SKIP Figure 2: fig2_architecture.png not built")
    else:
        log.append("MISS Figure 2 thumbnails not found")

    # ------------------------------------------------------- empty outlines
    for a, b in reversed(empty_shape_spans(xml, rels, spacers)):
        xml = xml[:a] + xml[b:]
        log.append("removed an empty outlined shape")

    # ---------------------------------------------------- floating figure 1
    # Figure 1 is anchored with wrapSquare, so its caption wraps into a
    # four-word-wide column beside a 7.3in-tall figure and is nearly
    # unreadable. A full-width figure has nothing to wrap around it; inline
    # placement is what the layout actually wants.
    n_unfloat = 0
    for a, b in reversed(de.top_level_paragraphs(xml)):
        frag = xml[a:b]
        if "<wp:anchor" not in frag or "wrapSquare" not in frag:
            continue
        rids = de.embedded_rids(frag)
        if len(rids) != 1:
            continue
        ext = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', frag)
        if not ext or int(ext.group(1)) / EMU_PER_IN < 4.0:
            continue                       # a small float can legitimately wrap
        rebuilt = ('<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="0" '
                   'w:before="0" w:line="240" w:lineRule="auto"/></w:pPr>'
                   + inline_picture(rids[0], int(ext.group(1)),
                                    int(ext.group(2)), "figure")
                   + "</w:p>")
        xml = xml[:a] + rebuilt + xml[b:]
        n_unfloat += 1
    if n_unfloat:
        log.append(f"made {n_unfloat} full-width figure(s) inline so the "
                   f"caption sits under the figure instead of beside it")

    # ------------------------------------------------- empty list items
    # An empty paragraph carrying <w:numPr> is not invisible: Word still
    # draws its bullet or number, so the reader sees a stray dash on a line
    # of its own. One sits directly under the Figure 3 caption.
    n_bullets = 0
    for a, b in reversed(de.top_level_paragraphs(xml)):
        seg = xml[a:b]
        if ("<w:numPr" in seg and not de.text_of(seg)
                and "<w:drawing" not in seg and "w:pict" not in seg):
            xml = xml[:a] + xml[b:]
            n_bullets += 1
    if n_bullets:
        log.append(f"removed {n_bullets} empty list item(s) that rendered "
                   f"as a stray bullet")

    # -------------------------------------------------------- blank runs
    # The manuscript carries long runs of empty paragraphs, inherited from
    # Version 6. They were not decoration: they reserved the space that
    # anchored, text-wrapped figures floated into. Once those figures are
    # inline they occupy their own space, so every reserved run became a
    # visible hole, up to 25 blank lines mid-page. Runs are collapsed to a
    # single blank rather than deleted outright, which keeps the paragraph
    # break the author put between two blocks. A blank that carries a page
    # break or the section properties is load-bearing and is left alone.
    xml, n_blank = collapse_blank_runs(xml)
    if n_blank:
        log.append(f"collapsed {n_blank} empty paragraph(s) left behind by "
                   f"figures that used to float")

    # ------------------------------------------------ duplicate callouts
    # Version 7 added a red callout for Figures 1 and 3 to guarantee each was
    # cited. Both were already cited: the sentence immediately above each
    # insertion names the same figure, so the reader meets the same pointer
    # twice in consecutive lines. Version 6 had exactly one callout apiece,
    # which is what stays.
    for lead in DUPLICATE_CALLOUTS:
        span = de.find_paragraph(xml, lambda t, L=lead: t.startswith(L))
        if span:
            a, b = span
            xml = xml[:a] + xml[b:]
            log.append(f"removed duplicate figure callout ({lead[:34]}...)")

    # ------------------------------------------------- figure/caption pairs
    # Two things keep a figure with its caption. The figure must be short
    # enough that both fit in the 9in text block, and every paragraph from the
    # figure down to the caption must carry keepNext, because the empty
    # paragraphs between them are page-break opportunities too. Version 7 had
    # neither, so each full-page figure was followed by a page whose only
    # content was an orphaned caption.
    # One cap, not two. Figure 2 is placed against FIG_MAX_H_IN above; if the
    # pass that resizes the other figures used a different ceiling the three
    # figures would print at different scales for no reason the reader can see.
    budget = min(FIG_MAX_H_IN, TEXT_H_IN - CAPTION_RESERVE_IN)
    spans = de.top_level_paragraphs(xml)

    def caption_index(start: int) -> int | None:
        """Index of the caption belonging to the figure at ``start``.

        Blank paragraphs in between are tolerated. Figure 1 has 36 of them:
        they were the space the text used to wrap around the floating image,
        and now that it is inline they are just an empty page. Any real prose
        before a caption means the figure is not captioned here at all.
        """
        for j in range(start + 1, len(spans)):
            t = de.body_text_of(xml[spans[j][0]:spans[j][1]])
            if not t:
                continue
            return j if re.match(r"^Figure\s+\d+\s*[.:]", t) else None
        return None

    n_fit = 0
    for i, (a, b) in reversed(list(enumerate(spans))):
        frag = xml[a:b]
        if "<w:drawing>" not in frag or not de.embedded_rids(frag):
            continue
        cap_i = caption_index(i)
        if cap_i is None:
            continue

        ext = re.search(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', frag)
        if not ext:
            continue
        cx, cy = int(ext.group(1)), int(ext.group(2))
        h_in = cy / EMU_PER_IN

        new = frag
        if h_in > budget:
            scale = budget / h_in
            ncx, ncy = int(cx * scale), int(cy * scale)
            new = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>',
                         f'<wp:extent cx="{ncx}" cy="{ncy}"/>', new)
            new = re.sub(r'<a:ext cx="\d+" cy="\d+"/>',
                         f'<a:ext cx="{ncx}" cy="{ncy}"/>', new)
            log.append(f"resized a figure from {h_in:.2f}in to "
                       f"{ncy / EMU_PER_IN:.2f}in tall so its caption fits "
                       f"on the same page")

        # Blank paragraphs between a figure and its caption were the space the
        # text used to wrap around a floating image. Inline, they only push the
        # caption onto the next page, so they go.
        blanks = [j for j in range(i + 1, cap_i)
                  if not de.body_text_of(xml[spans[j][0]:spans[j][1]])
                  and "<w:drawing>" not in xml[spans[j][0]:spans[j][1]]]
        for j in reversed(blanks):
            ja, jb = spans[j]
            xml = xml[:ja] + xml[jb:]
        if blanks:
            log.append(f"removed {len(blanks)} empty paragraph(s) that had "
                       f"separated a figure from its caption")

        # Bind whatever is left between the figure and the caption.
        for j in range(cap_i - 1, i, -1):
            if j in blanks:
                continue
            ja, jb = spans[j]
            xml = xml[:ja] + with_keep_next(xml[ja:jb]) + xml[jb:]
        xml = xml[:a] + with_keep_next(new) + xml[b:]
        n_fit += 1
    if n_fit:
        log.append(f"bound {n_fit} figure(s) to their captions with keepNext")

    # ------------------------------------------------------- orphan figures
    targets = drawing_paragraph_spans(xml, rels, ORPHAN_IMAGES)
    for a, b, names in sorted(targets, reverse=True):
        xml = xml[:a] + xml[b:]
        log.append(f"removed uncited trailing figure ({', '.join(sorted(names))})")

    doc.write_text(xml, encoding="utf-8")
    rels_path.write_text(rel_xml, encoding="utf-8")

    if dst.exists():
        dst.unlink()
    subprocess.run(["zip", "-q", "-r", "-X", str(dst), "."], cwd=work, check=True)
    return log


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/aayu/Downloads/Paper - Aayushya - Krish - Zaidi - Version 7.docx")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "/Users/aayu/Downloads/Paper - Aayushya - Krish - Zaidi - Version 7.1.docx")
    log = build(src, dst)
    print(f"wrote {dst}")
    for line in log:
        print("  -", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
