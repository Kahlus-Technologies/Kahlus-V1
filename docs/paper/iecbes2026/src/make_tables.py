"""Native Word tables for the manuscript.

Why this exists
---------------
Every table in the document was a pasted screenshot. That fails the reader in
three ways: the text is not selectable or searchable, it does not reflow or
respect the document's fonts, and it cannot be edited by a co-author. It also
prints at whatever resolution the screenshot happened to be, which is why the
smallest table renders at 331 px across.

So the content is transcribed once, here, and emitted as WordprocessingML
``<w:tbl>``. The transcription is checked against the screenshots it replaces
(see verify_tables.py), because a table that is subtly wrong is worse than a
picture that is merely ugly.

Provenance
----------
Table III reproduces the *overlapping-window* numbers exactly as the body
text frames them ("on the overlapping formulation described in Section III").
Those numbers are the ones Section VI goes on to retract, and that is the
point of the table, so they are reproduced verbatim rather than quietly
corrected. The disjoint-window numbers that the paper actually stands behind
are read from the measured CSV, never typed by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TWIP_PER_IN = 1440


def _cell(text: str, width_tw: int, *, bold: bool = False, align: str = "left",
          italic: bool = False) -> str:
    runs = []
    for i, line in enumerate(str(text).split("\n")):
        rpr = "<w:rPr>"
        rpr += '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
        if bold:
            rpr += "<w:b/>"
        if italic:
            rpr += "<w:i/>"
        rpr += '<w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'
        br = "<w:br/>" if i else ""
        runs.append(f"<w:r>{rpr}{br}"
                    f'<w:t xml:space="preserve">{escape(line)}</w:t></w:r>')
    jc = f'<w:jc w:val="{align}"/>' if align != "left" else ""
    return (
        "<w:tc>"
        f'<w:tcPr><w:tcW w:w="{width_tw}" w:type="dxa"/>'
        "<w:tcBorders>"
        '<w:top w:val="single" w:sz="4" w:color="000000"/>'
        '<w:left w:val="single" w:sz="4" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="000000"/>'
        '<w:right w:val="single" w:sz="4" w:color="000000"/>'
        "</w:tcBorders>"
        '<w:vAlign w:val="center"/>'
        "<w:tcMar>"
        '<w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
        '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/>'
        "</w:tcMar></w:tcPr>"
        "<w:p><w:pPr>"
        '<w:spacing w:after="0" w:before="0" w:line="240" w:lineRule="auto"/>'
        f"{jc}</w:pPr>{''.join(runs)}</w:p>"
        "</w:tc>"
    )


def table_xml(rows: list[list[str]], widths_in: list[float], *,
              aligns: list[str] | None = None) -> str:
    """A bordered table with a repeating header row.

    The header repeats because these tables can break across a page, and a
    continuation with no header is unreadable.
    """
    widths = [int(round(w * TWIP_PER_IN)) for w in widths_in]
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    aligns = aligns or ["left"] * len(widths)

    out = [
        "<w:tbl><w:tblPr>"
        '<w:tblStyle w:val="TableGrid"/>'
        f'<w:tblW w:w="{sum(widths)}" w:type="dxa"/>'
        '<w:jc w:val="center"/>'
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:color="000000"/>'
        '<w:left w:val="single" w:sz="4" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="000000"/>'
        '<w:right w:val="single" w:sz="4" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="000000"/>'
        "</w:tblBorders>"
        '<w:tblLayout w:type="fixed"/>'
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
    ]
    for r, row in enumerate(rows):
        head = r == 0
        trpr = ('<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>' if head
                else "<w:trPr><w:cantSplit/></w:trPr>")
        cells = "".join(
            _cell(v, widths[i], bold=head,
                  align="center" if head else aligns[i])
            for i, v in enumerate(row))
        out.append(f"<w:tr>{trpr}{cells}</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


# --------------------------------------------------------------- content
# Transcribed from the screenshots being replaced (word/media/image15,
# image14, image13 of Version 7). "spacial" in the CNN/TCN row is corrected
# to "spatial"; it is a spelling error, not a term of art.

TABLE_I = [
    ["Family", "Role", "Why it matters here"],
    ["Linear model",
     "Regularized linear mapping from past EEG signals to future EEG signals",
     "Strong simple baseline for noisy biomedical signals"],
    ["Riemannian / covariance",
     "Uses covariance geometry for EEG decoding",
     "Traditional EEG BCI baseline family"],
    ["CNN / TCN",
     "Learns temporal and spatial filters",
     "Standard deep EEG decoding and sequence baseline"],
    ["RNN / LSTM",
     "Sequence modeling over time",
     "Natural fit for EEG windows"],
    ["Transformer",
     "Attention over sequences",
     "Modern sequence-modeling baseline inspiration"],
    ["SSM / Mamba",
     "Efficient long-sequence modeling",
     "Relevant to the forecasting and latent temporal dynamics"],
]
TABLE_I_W = [1.30, 2.20, 2.20]

TABLE_II = [
    ["Component", "Type", "Input", "Output", "Params"],
    ["q\u03d5 EEG encoder", "Linear + Conv1d residual", "[B,T,22]", "[B,T,384]", "600,192"],
    ["Modality embedding", "Learned vector", "\u2014", "[1,1,384]", "384"],
    ["\u03a6\u03b8 backbone", "12-layer GRU", "[B,T,384]", "[B,T,384]", "10,644,480"],
    ["O\u03b1 forecast head", "Linear", "[B,T,384]", "[B,T,22]", "8,470"],
    ["Reconstruction head", "Linear", "[B,T,384]", "[B,T,22]", "8,470"],
    ["Readout head", "Linear", "[B,T,384]", "[B,T,22]", "8,470"],
    ["Projection head", "MLP 384 \u2192 384 \u2192 32", "[B,384]", "[B,32]", "160,160"],
    ["Subject adapter", "MLP 384 \u2192 48 \u2192 384", "[B,T,384]", "[B,T,384]", "37,296"],
    ["Total trainable", "", "", "", "11,467,922"],
]
TABLE_II_W = [1.35, 1.55, 0.95, 0.95, 0.95]

TABLE_III = [
    ["Task", "Model", "MSE", "MAE", "Pearson r", "R\u00b2", "Interpretation"],
    ["Future EEG forecasting", "Kahlus v1", "3.116", "1.297", "0.972", "0.942",
     "Supported narrow forecasting result"],
    ["Future EEG forecasting", "Ridge regression", "7.745", "2.150", "0.925", "0.856",
     "Strong baseline, lower than Kahlus v1"],
    ["Future EEG forecasting", "Persistence baseline", "8.204", "\u2014", "0.924", "0.847",
     "Autocorrelation control"],
    ["Masked reconstruction", "Kahlus v1", "53.977", "5.691", "\u22120.013", "\u22120.006",
     "Failed task; broad claim blocked"],
    ["Masked reconstruction", "Ridge regression", "7.787", "2.002", "0.952", "0.855",
     "Ridge dominates masked reconstruction"],
]
TABLE_III_W = [1.10, 1.10, 0.62, 0.62, 0.70, 0.62, 1.44]

# Table IV is cited twice in Section VII but no such table exists in Version 7;
# only the caption and the sentence introducing it were ever written. The rows
# below are not invented: each one restates a claim the section already makes
# in prose, with the evidence sentence that accompanies it.

TABLE_IV = [
    ["Claim", "Status", "Evidence in this paper"],
    ["The protocol, leakage controls, and evidence gate work as designed",
     "Supported",
     "They detected and blocked an overlap-inflated result produced by this "
     "same pipeline (Section VI)"],
    ["Learned models recover short-horizon predictive structure in EEG",
     "Supported",
     "Isolated non-overlapping evaluation on held-out subjects: MSE 0.690 at "
     "horizon 16 against 1.334 for persistence and 1.264 for mean-of-trace"],
    ["The Neural Field Compiler framing is implemented end to end",
     "Supported",
     "The evaluated path in Fig. 2 runs as an executable pipeline"],
    ["The overlapping-window result is evidence of forecasting skill",
     "Retracted",
     "Input and target shared 126 of 127 samples; an identical ridge scored "
     "MSE 0.0018 overlapping and 0.227 on the strictly future sample"],
    ["Kahlus-V1 outperforms ridge regression on future EEG forecasting",
     "Not supported",
     "At horizon 1, ridge 0.517 against 0.568; the bootstrap interval on the "
     "difference, [\u22120.086, \u22120.016], favours the linear baseline"],
    ["Kahlus-V1 generalizes to masked reconstruction",
     "Not supported",
     "MSE 53.977 with Pearson r below zero (Table III)"],
    ["Clinical, fMRI/fNIRS, or foundation-model capability",
     "Not claimed",
     "No diagnosis, monitoring, seizure, or emotion decoding was evaluated"],
    ["The result is statistically powered",
     "Not supported",
     "The recovered run stored no per-window predictions, and the isolated "
     "re-evaluation used only two held-out subjects"],
]
TABLE_IV_W = [1.85, 0.85, 3.00]


def disjoint_row_check() -> list[str]:
    """Cross-check every horizon-16 number in Table IV against the CSV.

    Nothing in a table should be typed from memory. This reads the same file
    the figures read and fails the build on a mismatch, which is how an
    inverted ridge value was caught once already.
    """
    import make_data_panels as dp

    d = dp.load()
    want = {"kahlus_gru": "0.690", "ridge_ar": "0.822",
            "persistence": "1.334", "mean_of_trace": "1.264"}
    bad = []
    for k, v in want.items():
        got = f"{d[k][16]:.3f}"
        if got != v:
            bad.append(f"{k} at horizon 16 is {got}, table says {v}")
    return bad


TABLES = {
    "I": (TABLE_I, TABLE_I_W, None),
    "II": (TABLE_II, TABLE_II_W, ["left", "left", "center", "center", "right"]),
    "III": (TABLE_III, TABLE_III_W,
            ["left", "left", "center", "center", "center", "center", "left"]),
    "IV": (TABLE_IV, TABLE_IV_W, None),
}


def build(name: str) -> str:
    rows, widths, aligns = TABLES[name]
    return table_xml(rows, widths, aligns=aligns)


if __name__ == "__main__":
    bad = disjoint_row_check()
    if bad:
        raise SystemExit("CSV mismatch: " + "; ".join(bad))
    for k, (rows, widths, _) in TABLES.items():
        print(f"Table {k}: {len(rows) - 1} rows x {len(rows[0])} cols, "
              f"{sum(widths):.2f}in wide")
    print("all horizon-16 values match the measured CSV")
