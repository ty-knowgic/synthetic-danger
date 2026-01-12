#!/usr/bin/env python3
# report.py
# Generate publishable Safety Hazard Report (XLSX + HTML) from hazards_enriched*.json
#
# Features:
# - Stable XLSX export with multiple sheets:
#   Hazards / Requirements / Verification / Traceability / Summary / SystemInput / TopRisks
# - Review workflow columns:
#   review_status / reviewer / review_notes / decision_date
# - Risk scoring (RPN) and sorting by risk by default
# - Executive summary (Top N risks) in HTML and in TopRisks sheet

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

TRUNC_MARKERS = ("...", "…")

# Simple weights (adjust later to match a customer standard)
SEVERITY_W = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
LIKELIHOOD_W = {"Low": 1, "Medium": 2, "High": 3}
DETECT_W = {"High": 1, "Medium": 2, "Low": 3}

DEFAULT_REVIEW_STATUS = "Draft"  # Draft/Reviewed/Accepted/Rejected


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (int, float, bool)):
        return str(s)
    return str(s).strip()


def join_lines(x: Any) -> str:
    """Join list[str] into newline-separated bullets; pass through strings."""
    if x is None:
        return ""
    if isinstance(x, list):
        items = [norm_text(i) for i in x if norm_text(i)]
        return "\n".join(f"- {i}" for i in items)
    return norm_text(x)


def has_trunc(s: str) -> bool:
    s2 = s.strip()
    if not s2:
        return False
    if any(s2.endswith(m) for m in TRUNC_MARKERS):
        return True
    if "..." in s2:
        return True
    return False


def make_ids(hazard_id: str) -> Tuple[str, str]:
    sr = f"SR-{hazard_id}-01"
    vr = f"VR-{hazard_id}-01"
    return sr, vr


def weight(map_: Dict[str, int], label: str, default: int) -> int:
    return map_.get(label, default)


def risk_score(severity: str, likelihood: str, detectability: str) -> int:
    s = weight(SEVERITY_W, severity, 2)
    l = weight(LIKELIHOOD_W, likelihood, 2)
    d = weight(DETECT_W, detectability, 2)
    return int(s * l * d)


def risk_band(score: int) -> str:
    # Interpretable bands; tweak later
    if score >= 30:
        return "Extreme"
    if score >= 18:
        return "High"
    if score >= 8:
        return "Medium"
    return "Low"


def to_rows(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    hazards = doc.get("hazards", [])

    for h in hazards:
        hid = norm_text(h.get("hazard_id"))
        sr_id, vr_id = make_ids(hid)

        sev = norm_text(h.get("severity"))
        lik = norm_text(h.get("likelihood"))
        det = norm_text(h.get("detectability"))
        rpn = risk_score(sev, lik, det)

        obs = join_lines(h.get("observables"))
        mit = join_lines(h.get("mitigations"))
        vrf = norm_text(h.get("verification_idea"))
        req = norm_text(h.get("safety_requirement"))

        # Review workflow fields (optional)
        review = h.get("review", {}) if isinstance(h.get("review"), dict) else {}

        rows.append(
            {
                "hazard_id": hid,
                "category": norm_text(h.get("category")),
                "subtype": norm_text(h.get("subtype")),
                "task_phase": norm_text(h.get("task_phase")),
                "severity": sev,
                "likelihood": lik,
                "detectability": det,
                "risk_rpn": rpn,
                "risk_band": risk_band(rpn),
                "affected_components": ", ".join([norm_text(x) for x in (h.get("affected_components") or [])]),
                "trigger_condition": norm_text(h.get("trigger_condition")),
                "primary_failure": norm_text(h.get("primary_failure")),
                "propagation_chain": join_lines(h.get("propagation_chain")),
                "final_impact": norm_text(h.get("final_impact")),
                "safety_requirement_id": norm_text(h.get("safety_requirement_id")) or sr_id,
                "verification_id": norm_text(h.get("verification_id")) or vr_id,
                "observables": obs,
                "mitigations": mit,
                "safety_requirement": req,
                "verification_idea": vrf,
                "flag_req_truncated": "YES" if has_trunc(req) else "",
                "flag_verif_truncated": "YES" if has_trunc(vrf) else "",
                "review_status": norm_text(review.get("status")) or DEFAULT_REVIEW_STATUS,
                "reviewer": norm_text(review.get("reviewer")),
                "review_notes": norm_text(review.get("notes")),
                "decision_date": norm_text(review.get("decision_date")),
            }
        )

    # Default sort: highest risk first, then severity, then phase, then id
    rows.sort(
        key=lambda r: (
            -int(r.get("risk_rpn", 0) or 0),
            -SEVERITY_W.get(r.get("severity", ""), 0),
            r.get("task_phase", ""),
            r.get("hazard_id", ""),
        )
    )
    return rows


def to_requirements_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "safety_requirement_id": r.get("safety_requirement_id", ""),
            "hazard_id": r.get("hazard_id", ""),
            "category": r.get("category", ""),
            "task_phase": r.get("task_phase", ""),
            "severity": r.get("severity", ""),
            "risk_rpn": r.get("risk_rpn", ""),
            "risk_band": r.get("risk_band", ""),
            "affected_components": r.get("affected_components", ""),
            "safety_requirement": r.get("safety_requirement", ""),
            "review_status": r.get("review_status", ""),
            "reviewer": r.get("reviewer", ""),
            "decision_date": r.get("decision_date", ""),
            "flag_req_truncated": r.get("flag_req_truncated", ""),
        }
        for r in rows
    ]


def to_verification_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "verification_id": r.get("verification_id", ""),
            "hazard_id": r.get("hazard_id", ""),
            "category": r.get("category", ""),
            "task_phase": r.get("task_phase", ""),
            "severity": r.get("severity", ""),
            "risk_rpn": r.get("risk_rpn", ""),
            "risk_band": r.get("risk_band", ""),
            "observables": r.get("observables", ""),
            "mitigations": r.get("mitigations", ""),
            "verification_idea": r.get("verification_idea", ""),
            "review_status": r.get("review_status", ""),
            "reviewer": r.get("reviewer", ""),
            "decision_date": r.get("decision_date", ""),
            "flag_verif_truncated": r.get("flag_verif_truncated", ""),
        }
        for r in rows
    ]


def to_traceability_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "hazard_id": r.get("hazard_id", ""),
            "safety_requirement_id": r.get("safety_requirement_id", ""),
            "verification_id": r.get("verification_id", ""),
            "severity": r.get("severity", ""),
            "risk_rpn": r.get("risk_rpn", ""),
            "risk_band": r.get("risk_band", ""),
            "task_phase": r.get("task_phase", ""),
            "category": r.get("category", ""),
            "review_status": r.get("review_status", ""),
        }
        for r in rows
    ]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cat = Counter(r["category"] for r in rows)
    sev = Counter(r["severity"] for r in rows)
    phase = Counter(r["task_phase"] for r in rows)
    band = Counter(r["risk_band"] for r in rows)

    trunc_req = sum(1 for r in rows if r.get("flag_req_truncated") == "YES")
    trunc_ver = sum(1 for r in rows if r.get("flag_verif_truncated") == "YES")

    return {
        "total_hazards": len(rows),
        "category_distribution": dict(cat),
        "severity_distribution": dict(sev),
        "task_phase_top15": dict(phase.most_common(15)),
        "risk_band_distribution": dict(band),
        "truncation_flags": {"safety_requirement_truncated": trunc_req, "verification_idea_truncated": trunc_ver},
    }


def top_risks(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return rows[: max(0, n)]


def write_html(out_path: Path, doc: Dict[str, Any], rows: List[Dict[str, Any]], summary: Dict[str, Any], topn: int) -> None:
    system_input = norm_text(doc.get("system_input"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#039;")
        )

    def dict_to_ul(d: Dict[str, Any]) -> str:
        items = []
        for k, v in d.items():
            items.append(f"<li><b>{esc(str(k))}</b>: {esc(str(v))}</li>")
        return "<ul>" + "".join(items) + "</ul>"

    def table_html(title: str, anchor: str, cols: List[str], data_rows: List[Dict[str, Any]]) -> str:
        trs = []
        for r in data_rows:
            tds = []
            for c in cols:
                val = norm_text(r.get(c))
                tds.append(f"<td><pre>{esc(val)}</pre></td>")
            trs.append("<tr>" + "".join(tds) + "</tr>")
        return f"""
<div class="card" id="{anchor}">
  <h2>{esc(title)}</h2>
  <table>
    <thead>
      <tr>{''.join([f'<th><pre>{esc(c)}</pre></th>' for c in cols])}</tr>
    </thead>
    <tbody>
      {''.join(trs)}
    </tbody>
  </table>
</div>
""".strip()

    toc = """
<div class="card" id="toc">
  <h2>Contents</h2>
  <ul>
    <li><a href="#summary">Summary</a></li>
    <li><a href="#toprisks">Top risks</a></li>
    <li><a href="#hazards">Hazards</a></li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#verification">Verification</a></li>
    <li><a href="#traceability">Traceability</a></li>
  </ul>
</div>
""".strip()

    summary_card = f"""
<div class="card" id="summary">
  <h2>Summary</h2>
  <div><b>Generated</b>: {esc(now)}</div>
  <div><b>Total hazards</b>: {summary['total_hazards']}</div>
  <h3>Risk bands</h3>
  {dict_to_ul(summary['risk_band_distribution'])}
  <h3>Category distribution</h3>
  {dict_to_ul(summary['category_distribution'])}
  <h3>Severity distribution</h3>
  {dict_to_ul(summary['severity_distribution'])}
  <h3>Top task phases (15)</h3>
  {dict_to_ul(summary['task_phase_top15'])}
  <h3>Truncation flags</h3>
  {dict_to_ul(summary['truncation_flags'])}
</div>
""".strip()

    system_card = f"""
<div class="card" id="system">
  <h2>System Input</h2>
  <pre>{esc(system_input)}</pre>
</div>
""".strip()

    top_cols = [
        "hazard_id", "risk_rpn", "risk_band", "severity", "likelihood", "detectability",
        "category", "task_phase", "affected_components", "trigger_condition", "final_impact",
        "safety_requirement_id", "verification_id", "review_status",
    ]

    hazards_cols = [
        "hazard_id", "risk_rpn", "risk_band",
        "category", "subtype", "task_phase",
        "severity", "likelihood", "detectability",
        "affected_components",
        "trigger_condition", "primary_failure", "final_impact",
        "safety_requirement_id", "safety_requirement",
        "verification_id", "verification_idea",
        "observables", "mitigations",
        "review_status", "reviewer", "decision_date", "review_notes",
        "flag_req_truncated", "flag_verif_truncated",
    ]

    req_cols = [
        "safety_requirement_id", "hazard_id",
        "risk_rpn", "risk_band",
        "category", "task_phase", "severity",
        "affected_components",
        "safety_requirement",
        "review_status", "reviewer", "decision_date",
        "flag_req_truncated",
    ]

    ver_cols = [
        "verification_id", "hazard_id",
        "risk_rpn", "risk_band",
        "category", "task_phase", "severity",
        "observables", "mitigations", "verification_idea",
        "review_status", "reviewer", "decision_date",
        "flag_verif_truncated",
    ]

    tr_cols = [
        "hazard_id", "safety_requirement_id", "verification_id",
        "risk_rpn", "risk_band",
        "severity", "task_phase", "category",
        "review_status",
    ]

    css = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; }
.card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }
pre { white-space: pre-wrap; word-wrap: break-word; margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; vertical-align: top; }
th { background: #f6f6f6; position: sticky; top: 0; z-index: 1; }
th pre { font-size: 12px; }
td pre { padding: 8px; }
a { color: #0b57d0; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #ccc; margin-left: 8px; font-size: 12px; }
</style>
""".strip()

    top_rows = top_risks(rows, topn)

    body = "\n".join(
        [
            toc,
            summary_card,
            system_card,
            table_html(f"Top risks (Top {topn})", "toprisks", top_cols, top_rows),
            table_html("Hazards (sorted by risk)", "hazards", hazards_cols, rows),
            table_html("Requirements", "requirements", req_cols, to_requirements_rows(rows)),
            table_html("Verification", "verification", ver_cols, to_verification_rows(rows)),
            table_html("Traceability", "traceability", tr_cols, to_traceability_rows(rows)),
        ]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Synthetic Danger Report</title>
{css}
</head>
<body>
<h1>Synthetic Danger Report <span class="badge">risk-sorted</span></h1>
{body}
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")


def write_xlsx(out_path: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any], doc: Dict[str, Any], topn: int) -> None:
    if pd is None:
        raise RuntimeError("pandas is not available. Install with: pip install pandas openpyxl")

    df = pd.DataFrame(rows)
    req_df = pd.DataFrame(to_requirements_rows(rows))
    ver_df = pd.DataFrame(to_verification_rows(rows))
    tr_df = pd.DataFrame(to_traceability_rows(rows))
    top_df = pd.DataFrame(top_risks(rows, topn))

    cat_df = pd.DataFrame(list(summary["category_distribution"].items()), columns=["category", "count"]).sort_values(
        "count", ascending=False
    )
    sev_df = pd.DataFrame(list(summary["severity_distribution"].items()), columns=["severity", "count"]).sort_values(
        "count", ascending=False
    )
    band_df = pd.DataFrame(list(summary["risk_band_distribution"].items()), columns=["risk_band", "count"]).sort_values(
        "count", ascending=False
    )
    ph_df = pd.DataFrame(list(summary["task_phase_top15"].items()), columns=["task_phase", "count"])
    trunc_df = pd.DataFrame(list(summary["truncation_flags"].items()), columns=["flag", "count"])

    meta_df = pd.DataFrame(
        [
            ["generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["total_hazards", summary["total_hazards"]],
            ["top_n", topn],
            ["sort", "risk_rpn desc"],
        ],
        columns=["key", "value"],
    )
    sys_df = pd.DataFrame([norm_text(doc.get("system_input"))], columns=["system_input"])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:  # type: ignore
        top_df.to_excel(writer, sheet_name="TopRisks", index=False)
        df.to_excel(writer, sheet_name="Hazards", index=False)
        req_df.to_excel(writer, sheet_name="Requirements", index=False)
        ver_df.to_excel(writer, sheet_name="Verification", index=False)
        tr_df.to_excel(writer, sheet_name="Traceability", index=False)

        meta_df.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
        band_df.to_excel(writer, sheet_name="Summary", index=False, startrow=5)
        cat_df.to_excel(writer, sheet_name="Summary", index=False, startrow=5 + len(band_df) + 3)
        sev_df.to_excel(writer, sheet_name="Summary", index=False, startrow=5 + len(band_df) + len(cat_df) + 6)
        ph_df.to_excel(writer, sheet_name="Summary", index=False, startrow=5 + len(band_df) + len(cat_df) + len(sev_df) + 9)
        trunc_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
            startrow=5 + len(band_df) + len(cat_df) + len(sev_df) + len(ph_df) + 12,
        )

        sys_df.to_excel(writer, sheet_name="SystemInput", index=False)

        wb = writer.book

        def fmt_sheet(name: str, wrap_cols: List[str], width_map: Dict[str, int]) -> None:
            ws = wb[name]
            ws.freeze_panes = "A2"
            header = [cell.value for cell in ws[1]]
            col_idx = {n: i + 1 for i, n in enumerate(header)}

            try:
                from openpyxl.styles import Alignment, Font  # type: ignore

                wrap = Alignment(wrap_text=True, vertical="top")
                top = Alignment(vertical="top")
                bold = Font(bold=True)

                for cell in ws[1]:
                    cell.font = bold

                for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
                    for cell in row:
                        col_name = header[cell.col_idx - 1]
                        cell.alignment = wrap if col_name in wrap_cols else top
            except Exception:
                pass

            for name2, idx in col_idx.items():
                letter = ws.cell(row=1, column=idx).column_letter
                ws.column_dimensions[letter].width = width_map.get(name2, 40)

        fmt_sheet(
            "TopRisks",
            wrap_cols=["trigger_condition", "final_impact", "safety_requirement", "verification_idea", "review_notes", "observables", "mitigations", "propagation_chain", "primary_failure"],
            width_map={
                "hazard_id": 14,
                "risk_rpn": 10,
                "risk_band": 10,
                "category": 14,
                "subtype": 16,
                "task_phase": 18,
                "severity": 10,
                "likelihood": 10,
                "detectability": 12,
                "affected_components": 30,
                "safety_requirement_id": 18,
                "verification_id": 18,
                "review_status": 14,
                "reviewer": 14,
                "decision_date": 14,
            },
        )

        fmt_sheet(
            "Hazards",
            wrap_cols=[
                "trigger_condition",
                "primary_failure",
                "propagation_chain",
                "final_impact",
                "observables",
                "mitigations",
                "safety_requirement",
                "verification_idea",
                "review_notes",
            ],
            width_map={
                "hazard_id": 14,
                "risk_rpn": 10,
                "risk_band": 10,
                "category": 14,
                "subtype": 16,
                "task_phase": 18,
                "severity": 10,
                "likelihood": 10,
                "detectability": 12,
                "affected_components": 30,
                "safety_requirement_id": 18,
                "verification_id": 18,
                "review_status": 14,
                "reviewer": 14,
                "decision_date": 14,
                "flag_req_truncated": 18,
                "flag_verif_truncated": 18,
            },
        )

        fmt_sheet(
            "Requirements",
            wrap_cols=["safety_requirement"],
            width_map={
                "safety_requirement_id": 18,
                "hazard_id": 14,
                "risk_rpn": 10,
                "risk_band": 10,
                "category": 14,
                "task_phase": 18,
                "severity": 10,
                "affected_components": 30,
                "review_status": 14,
                "reviewer": 14,
                "decision_date": 14,
                "flag_req_truncated": 18,
            },
        )

        fmt_sheet(
            "Verification",
            wrap_cols=["verification_idea", "observables", "mitigations"],
            width_map={
                "verification_id": 18,
                "hazard_id": 14,
                "risk_rpn": 10,
                "risk_band": 10,
                "category": 14,
                "task_phase": 18,
                "severity": 10,
                "review_status": 14,
                "reviewer": 14,
                "decision_date": 14,
                "flag_verif_truncated": 18,
            },
        )

        fmt_sheet(
            "Traceability",
            wrap_cols=[],
            width_map={
                "hazard_id": 14,
                "safety_requirement_id": 18,
                "verification_id": 18,
                "risk_rpn": 10,
                "risk_band": 10,
                "severity": 10,
                "task_phase": 18,
                "category": 14,
                "review_status": 14,
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate XLSX + HTML report from hazards_enriched*.json")
    ap.add_argument("--input", default="outputs/hazards_enriched.json")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--basename", default="report")
    ap.add_argument("--no-xlsx", action="store_true")
    ap.add_argument("--top", type=int, default=20, help="Number of top risks for executive summary")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    doc = load_json(in_path)
    rows = to_rows(doc)
    summary = summarize(rows)

    html_path = outdir / f"{args.basename}.html"
    write_html(html_path, doc, rows, summary, args.top)
    print(f"[ok] wrote: {html_path}")

    if not args.no_xlsx:
        xlsx_path = outdir / f"{args.basename}.xlsx"
        try:
            write_xlsx(xlsx_path, rows, summary, doc, args.top)
            print(f"[ok] wrote: {xlsx_path}")
        except Exception as e:
            print(f"[warn] failed to write xlsx: {type(e).__name__}: {e}")
            print("[warn] You can still use the HTML report, or install deps: pip install pandas openpyxl")

    print("=== REPORT QA ===")
    print(f"input_used: {in_path}")
    print(f"total_hazards: {summary['total_hazards']}")
    print(f"risk_band_distribution: {summary['risk_band_distribution']}")
    print(f"truncation_flags: {summary['truncation_flags']}")


if __name__ == "__main__":
    main()