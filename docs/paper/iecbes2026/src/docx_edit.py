"""Minimal WordprocessingML surgery helpers.

The one thing worth knowing
---------------------------
``re.findall(r"<w:p[ >].*?</w:p>", xml)`` is wrong, and wrong in a way that
reads as right. Paragraphs nest: a floating shape carries a ``<w:txbxContent>``
containing its own ``<w:p>``. The non-greedy match therefore stops at the
*inner* closing tag and hands back a truncated fragment. An earlier pass over
this document concluded that half the figures had no image data at all, purely
because of that regex.

So paragraphs are found by scanning with a depth counter instead. Everything
else in this module is small enough to read.
"""

from __future__ import annotations

import re

_P_TOKEN = re.compile(r"<w:p\b[^>]*?(/?)>|</w:p>")

# `<w:t[^>]*>` also matches `<w:txbxContent>`, because `[^>]*` happily eats
# "xbxContent". The word boundary is not optional here.
_TEXT_RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)


def top_level_paragraphs(xml: str) -> list[tuple[int, int]]:
    """Spans of every paragraph that is not nested inside another one."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for m in _P_TOKEN.finditer(xml):
        token = m.group(0)
        if token.startswith("</"):
            depth -= 1
            if depth == 0:
                spans.append((start, m.end()))
        elif m.group(1) == "/":          # self-closing <w:p/>
            if depth == 0:
                spans.append((m.start(), m.end()))
        else:
            if depth == 0:
                start = m.start()
            depth += 1
    return spans


def paragraphs(xml: str) -> list[str]:
    return [xml[a:b] for a, b in top_level_paragraphs(xml)]


def text_of(xml_fragment: str) -> str:
    """Visible text of a fragment, including any text inside floating shapes."""
    return re.sub(r"\s+", " ", "".join(_TEXT_RUN.findall(xml_fragment))).strip()


def text_runs(xml_fragment: str) -> list[str]:
    """Every <w:t> run in order, unmerged.

    visible_text() joins runs, which is right for matching a paragraph but
    wrong for recovering one label out of a drawing that holds several.
    """
    return _TEXT_RUN.findall(xml_fragment)


def body_text_of(paragraph: str) -> str:
    """Visible text excluding anything inside a floating shape's text box.

    Used when matching body prose, so a stray caption living inside a drawing
    cannot masquerade as the paragraph that follows it.
    """
    stripped = re.sub(r"<w:txbxContent>.*?</w:txbxContent>", "", paragraph, flags=re.S)
    return text_of(stripped)


def embedded_rids(xml_fragment: str) -> list[str]:
    return re.findall(r'r:embed="([^"]+)"', xml_fragment)


def rel_map(rels_xml: str) -> dict[str, str]:
    return dict(re.findall(r'Id="([^"]+)"[^>]*Target="media/([^"]+)"', rels_xml))


def find_paragraph(xml: str, predicate, *, use_body_text: bool = True):
    """First paragraph whose text satisfies ``predicate``; span or None."""
    reader = body_text_of if use_body_text else text_of
    for a, b in top_level_paragraphs(xml):
        if predicate(reader(xml[a:b])):
            return a, b
    return None
