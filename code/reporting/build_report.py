"""
Build report/report.pdf from report/report.md plus generated appendices.

Steps
  1. Traceability: every [ID] cited in report.md must exist in report/evidence_index.csv and
     every index row must be cited; every value must match its file
     (code/reporting/check_report_numbers.py).
  2. Generate appendices from the data files:
       A  evidence index (number -> file -> locator)
       B  AI assistance
       C  all reviewed disagreements (from the signed adjudication tables)
       D  supporting-file index
  3. Render with pandoc (Markdown -> HTML) and headless Chromium (HTML -> PDF).

Requires pandoc and Chromium/Chrome (env var CHROME= to point at one; wkhtmltopdf is a fallback).
Not part of the data pipeline.
Usage (from the repository root):
    python -m code.reporting.build_report
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re
import shutil
import subprocess

from code.reporting import check_report_numbers

REPORT = Path("report")
CSS = """
@page { size: A4; margin: 15mm 15mm 15mm 15mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10.4pt; line-height: 1.38; color: #1f1f1e; }
h1 { font-size: 12.5pt; margin: 9px 0 3px; border-bottom: 1px solid #ccc; padding-bottom: 1px; }
h1.title { font-size: 15pt; border: none; margin-bottom: 0; }
p.subtitle { font-size: 10pt; margin-top: 2px; color: #444; }
p { margin: 3px 0 5px; text-align: justify; }
table { border-collapse: collapse; margin: 4px 0 6px; font-size: 9.4pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2px 5px; text-align: left; vertical-align: top; }
th { background: #eef2f7; }
code { font-size: 8.6pt; background: #f4f4f4; padding: 0 2px; }
figure { margin: 4px 0; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.4pt; color: #444; }
.appendix { page-break-before: always; }
.small table, .small { font-size: 7.6pt; }
ol, ul { margin: 2px 0 4px 18px; padding-left: 4px; }
"""


def _rows(p: str) -> list[dict]:
    return list(csv.DictReader(Path(p).open(encoding="utf-8")))


def check_citations(text: str) -> None:
    cited = set(re.findall(r"\b([A-F]\d{1,2})\b", " ".join(re.findall(r"\[([A-F0-9,\s–-]+)\]", text))))
    # expand ranges like C3–C5 / A3-A5
    for a, b in re.findall(r"\b([A-F]\d{1,2})[–-]([A-F]\d{1,2})\b", text):
        if a[0] == b[0]:
            cited |= {f"{a[0]}{i}" for i in range(int(a[1:]), int(b[1:]) + 1)}
    index = {r["id"] for r in _rows("report/evidence_index.csv")}
    missing, unused = sorted(cited - index), sorted(index - cited)
    if missing:
        raise SystemExit(f"Cited IDs missing from evidence_index.csv: {missing}")
    if unused:
        print(f"Note: indexed but not cited in the text (kept as supporting numbers): {unused}")


def appendices() -> str:
    out = ['\n<div class="appendix"></div>\n\n# Appendix A: Evidence index\n',
           "Every number in the report, the file that holds it, and how it is located. "
           "`python -m code.reporting.check_report_numbers` re-reads them all; the result is in "
           "`supporting_files/report_traceability.csv`.\n",
           '<div class="small">\n', "| ID | Claim | Value | File | Locator |", "|---|---|---|---|---|"]
    for r in _rows("report/evidence_index.csv"):
        val = r["value"] if len(r["value"]) < 20 else r["value"][:16] + "…"
        out.append(f"| {r['id']} | {r['claim']} | {val} | `{r['file']}` | `{r['locator']}` |")
    out.append("\n</div>\n")

    out += ['\n# Appendix B: Use of AI assistance\n',
            "**Assistant.** Claude (claude.ai agentic workspace, configured model `claude-opus-5-5`) was used on 22 Oct 2026 for Parts C (analysis), D, E and F. "
            "The exact prompts and the files each one produced are in `prompts/ai_assistance/claude_session_2026-10-22.md`. "
            "The assistant read the repository, wrote and ran code in a sandbox copy, and returned stage zips. "
            "The team unzipped them, reran every script on their own machine and committed the results.\n",
            "**What the assistant did:**\n",
            "- wrote the analysis, augmentation, active-learning, split, DVC and reporting code, with tests;\n",
            "- drafted adjudication proposals from trace evidence;\n",
            "- drafted guidelines v2 and these documents.\n",
            "**What humans did:**\n",
            "- every annotation (rounds 1 and 2);\n",
            "- confirmation or change of every adjudication (ADJ1 changed 2 of 19 round-2 proposals);\n",
            "- the paraphrase spot check;\n",
            "- every command run and commit.\n",
            "**Prompts sent to models by the code** are in `prompts/model_prompts/` and are verified against the code by `tests/test_prompts.py`. "
            "Any other assistant the team used must be listed in `prompts/ai_assistance/OTHER_TOOLS.md`.\n"]

    out += ['\n<div class="appendix"></div>\n\n# Appendix C: Disagreements reviewed\n',
            "All 27 items where the two annotators disagreed on at least one level. Shown for each: "
            "the A1 and A2 labels (outcome / category / step), the evidence from the trace, and the final decision. "
            "Full justifications and rule references: `supporting_files/agreement/round*/adjudication_round*.csv`.\n",
            '<div class="small">\n', "| Round | Run | Task | A1 | A2 | Trace evidence (abridged) | Final (ADJ1) |", "|---|---|---|---|---|---|---|"]
    ab = lambda r, k: f"{r[k + '_outcome']} / {r[k + '_category']} / {r[k + '_step'] if r[k + '_step'] not in ('', '-1') else '–'}"
    for rnd in ("round1", "round2"):
        for r in _rows(f"supporting_files/agreement/{rnd}/adjudication_{rnd}.csv"):
            ev = r["trace_evidence"].replace("|", "/")
            ev = ev if len(ev) <= 260 else ev[:257] + "…"
            fin = f"{r['proposed_outcome']} / {r['proposed_category']} / {r['proposed_first_error_step'] or '–'}"
            out.append(f"| {rnd[-1]} | {r['run_id'][:8]} | {r['task_id']} | {ab(r, 'A1')} | {ab(r, 'A2')} | {ev} | {fin} |")
    out.append("\n</div>\n")

    out += ['\n# Appendix D: Supporting files\n',
            "| Evidence | Location |", "|---|---|",
            "| Raw Label Studio exports (unchanged), by round and annotator | `supporting_files/labels/`, `supporting_files/labels/round2/` |",
            "| Agreement calculations, confusion matrices, disagreements | `supporting_files/agreement/round1/`, `round2/`, `pre_post_comparison.*` |",
            "| Validation outputs on raw, processed, corrupted and augmented data | `supporting_files/validation/` |",
            "| Grader v1 vs v2 | `supporting_files/grader_v2/`, `data/processed/regrade_v2.csv` |",
            "| Augmentation checks, examples, rejected candidates, spot check | `supporting_files/augmentation/` |",
            "| Active-learning curves, plot data, selection manifests | `supporting_files/active_learning/`, `data/active_learning/` |",
            "| Detector evaluation | `supporting_files/detectors/` |",
            "| Split leakage report | `supporting_files/splits/leakage_report.json` |",
            "| Dataset statistics; report traceability | `supporting_files/dataset_stats.json`, `supporting_files/report_traceability.csv` |",
            "| Test log | `supporting_files/logs/` |",
            "| 2-minute screen recording of the pipeline | `supporting_files/recording/` (**to add**) |"]
    return "\n".join(out) + "\n"


def main() -> None:
    text = (REPORT / "report.md").read_text(encoding="utf-8")
    check_citations(text)
    check_report_numbers.main()
    full = REPORT / "report_full.md"
    full.write_text(text + appendices(), encoding="utf-8")
    css = REPORT / "report.css"
    css.write_text(CSS, encoding="utf-8")
    html_out = REPORT / "report.html"
    subprocess.run(["pandoc", str(full), "-s", "--css", "report.css", "--metadata", "lang=en",
                    "--resource-path", str(REPORT), "-o", str(html_out)], check=True)
    chrome = os.environ.get("CHROME") or next((c for c in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")
                                                if shutil.which(c)), None)
    pdf = (REPORT / "report.pdf").resolve()
    if chrome:
        subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", html_out.resolve().as_uri()], check=True, capture_output=True)
    else:  # fallback: page CSS is only partly honoured by wkhtmltopdf
        subprocess.run(["wkhtmltopdf", "--quiet", "--enable-local-file-access", "-s", "A4",
                        str(html_out), str(pdf)], check=True)
    for tmp in (html_out, css, full):
        tmp.unlink()
    print("wrote report/report.pdf")


if __name__ == "__main__":
    if not shutil.which("pandoc"):
        raise SystemExit("pandoc is required (apt install pandoc); plus Chromium/Chrome (or wkhtmltopdf as a fallback)")
    main()
