# Synthetic Danger (PoC)

Synthetic Danger generates **draft hazards**, **safety requirements**, and **verification ideas** from structured system definitions, enabling faster robotic safety reviews with LLM support.

> ⚠️ Disclaimer  
> This tool generates *draft artifacts* to accelerate safety reviews.  
> It does **not** certify compliance with any safety standard (e.g., ISO 10218, ISO 13849, IEC 61508).  
> Final validation and approval must be performed by qualified safety engineers.

---

## What this tool does

- Generates structured hazard candidates by category
- Adds draft safety requirements and verification ideas
- Normalizes components and task phases
- Repairs truncated LLM outputs automatically
- Exports review‑ready HTML and Excel reports

The intent is to shift safety work from *blank‑page brainstorming* to *review and decision making*.

---

## Requirements

- macOS (tested)
- Python 3.11+
- OpenAI API key

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install openai python-dotenv jsonschema pandas openpyxl pyyaml
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=YOUR_API_KEY
```

Optional:
```bash
OPENAI_MODEL=gpt-4.1-mini
TARGET_PER_CATEGORY=15
```

---

## Input format (YAML – recommended)

Edit:

```
templates/system_input.yaml
```

Validate before execution:

```bash
python3 validate_input.py --yaml templates/system_input.yaml
```

Run with validation:

```bash
python3 run.py --mode yaml --yaml templates/system_input.yaml --validate
```

Output:

```
outputs/hazards.json
```

---

## Full processing pipeline

```bash
python3 analyze.py
python3 normalize.py
python3 enrich.py --resume
python3 repair_truncation.py
python3 report.py --input outputs/hazards_enriched_repaired.json --basename report_latest
```

Generated files:

- outputs/report_latest.html
- outputs/report_latest.xlsx

Open on macOS:

```bash
open outputs/report_latest.html
open outputs/report_latest.xlsx
```

---

## Review workflow

The Excel report includes review fields:

- review_status (Draft / Reviewed / Accepted / Rejected)
- reviewer
- review_notes
- decision_date

This enables direct use in safety review meetings.

---

## TXT input mode (legacy)

If you prefer free‑text input:

```bash
python3 run.py --mode txt --txt data/system.txt
```

---

## Troubleshooting

### YAML file not found

Run from repository root:

```bash
ls -la templates
```

Or specify absolute path:

```bash
python3 run.py --mode yaml --yaml /full/path/system_input.yaml
```

### PyYAML not installed

```bash
pip install pyyaml
```

### Output fields truncated

Run:

```bash
python3 repair_truncation.py
```

Then regenerate the report from `hazards_enriched_repaired.json`.

---

## Scope and limitations

This project is a proof‑of‑concept. It does not replace formal safety engineering processes.

Typical future extensions:

- stricter input schema validation
- enrich‑time truncation recovery
- phase balancing
- export for FMEA / FTA workflows

---

## License

Safety Danger is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.

- You may use, modify, and distribute this project, including in commercial settings.
- If you modify and distribute files covered by MPL-2.0, you must make those modified files available under MPL-2.0.
- You may keep separate proprietary code (e.g., plugins, integrations, or enterprise extensions) closed-source, as long as it is not a modification of MPL-covered files.

This project is a **thought-extension pipeline for human safety design**, not a compliance evidence generator and not a formal-methods/verification tool.

## Commercial extensions

We may offer paid extensions (e.g., review-assist features, quality-control layers, or enterprise integrations) under separate commercial terms.

The open-source core remains MPL-2.0.

## Disclaimer

Safety Danger provides generated artifacts to support **human review and decision-making**.
It does not guarantee correctness, completeness, or safety outcomes.
You are responsible for verification, validation, and real-world safety decisions.
