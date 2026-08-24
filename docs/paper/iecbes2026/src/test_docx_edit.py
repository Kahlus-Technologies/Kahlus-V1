"""Tests for the WordprocessingML surgery.

Run: python3 test_docx_edit.py

These are the checks that caught real bugs while building Version 7.1:

  * ``<w:t[^>]*>`` silently matches ``<w:txbxContent>``, so every empty text
    box looked like it contained text and the empty outlined rectangle on the
    Figure 2 page survived a removal pass that reported success.
  * A non-greedy ``<w:p>...</w:p>`` match truncates at the first *nested*
    closing tag, which is how an earlier audit concluded the document had no
    image data at all.
  * ``with_keep_next`` on a self-closing ``<w:p/>`` produced ``<w:p/><w:pPr>
    ...``, which is not well formed. Word would have refused the file.
"""

from __future__ import annotations

import re
import sys
import xml.dom.minidom as md
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import docx_edit as de              # noqa: E402
from make_v71_docx import with_keep_next  # noqa: E402

NS = (' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
      ' xmlns:w14="urn:x"')

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ------------------------------------------------------------ paragraphs
NESTED = (
    "<w:p><w:r><w:t>outer</w:t></w:r>"
    "<w:drawing><w:txbxContent>"
    "<w:p><w:r><w:t>inside a text box</w:t></w:r></w:p>"
    "</w:txbxContent></w:drawing></w:p>"
    "<w:p><w:r><w:t>second</w:t></w:r></w:p>"
)
paras = de.paragraphs(NESTED)
check("nested paragraphs are not split", len(paras) == 2, f"got {len(paras)}")
check("outer paragraph is whole", paras[0].endswith("</w:drawing></w:p>"))
check("nested text is visible to text_of",
      de.text_of(paras[0]) == "outerinside a text box")
check("nested text is hidden from body_text_of",
      de.body_text_of(paras[0]) == "outer")

check("self-closing paragraphs are counted",
      len(de.paragraphs("<w:p/><w:p><w:r><w:t>a</w:t></w:r></w:p>")) == 2)

# --------------------------------------------------------- txbxContent bug
EMPTY_BOX = ("<w:p><w:drawing><w:txbxContent><w:p><w:r>"
             '<w:t xml:space="preserve"></w:t></w:r></w:p>'
             "</w:txbxContent></w:drawing></w:p>")
check("an empty text box reads as empty", de.text_of(EMPTY_BOX) == "",
      f"got {de.text_of(EMPTY_BOX)!r}")

FULL_BOX = ("<w:p><w:drawing><w:txbxContent><w:p><w:r>"
            "<w:t>Future operators</w:t></w:r></w:p>"
            "</w:txbxContent></w:drawing></w:p>")
check("a text box with content is not empty",
      de.text_of(FULL_BOX) == "Future operators")

# ------------------------------------------------------------- keepNext
KEEP_CASES = {
    "no pPr": "<w:p><w:r><w:t>x</w:t></w:r></w:p>",
    "with pPr": '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
                "<w:r><w:t>x</w:t></w:r></w:p>",
    "with attributes": '<w:p w:rsidR="0" w14:paraId="AB">'
                       "<w:r><w:t>x</w:t></w:r></w:p>",
    "already keeps": "<w:p><w:pPr><w:keepNext/></w:pPr>"
                     "<w:r><w:t>x</w:t></w:r></w:p>",
    "leading whitespace": '<w:p>\n  <w:pPr><w:jc w:val="left"/></w:pPr>'
                          "<w:r><w:t>x</w:t></w:r></w:p>",
}
for label, src in KEEP_CASES.items():
    out = with_keep_next(src)
    try:
        doc = md.parseString(
            re.sub(r"^<w:p\b([^>]*)>", lambda m: "<w:p" + m.group(1) + NS + ">",
                   out, count=1))
        kids = [n for n in doc.documentElement.childNodes if n.nodeType == 1]
        ppr = [n.tagName for n in kids[0].childNodes if n.nodeType == 1]
        ok = (out.count("<w:keepNext/>") == 1
              and kids[0].tagName == "w:pPr" and ppr[0] == "w:keepNext")
        check(f"keepNext, {label}", ok, f"pPr={ppr}")
    except Exception as exc:                     # noqa: BLE001
        check(f"keepNext, {label}", False, str(exc))

out = with_keep_next("<w:p/>")
check("keepNext leaves a self-closing paragraph alone", out == "<w:p/>",
      f"got {out!r}")

# ---------------------------------------------------------------- summary
print()
if failures:
    print(f"{len(failures)} failing: " + ", ".join(failures))
    raise SystemExit(1)
print("all checks pass")
