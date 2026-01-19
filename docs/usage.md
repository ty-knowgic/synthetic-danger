# Usage Guide

## Execution Modes

### YAML Mode (Recommended)

Edit `templates/system_input.yaml` and run:

```bash
python3 run.py --mode yaml --yaml templates/system_input.yaml --validate
```

### TXT Mode (Legacy)

If you prefer free-text:

```bash
python3 run.py --mode txt --txt data/system.txt
```

## Processing Pipeline

The full pipeline consists of several steps to ensure high-quality, normalized output:

1. **Analyze**: `python3 analyze.py` (Initial hazard identification)
2. **Normalize**: `python3 normalize.py` (Collapses similar components/phases)
3. **Enrich**: `python3 enrich.py --resume` (Adds requirements and verification ideas)
4. **Repair**: `python3 repair_truncation.py` (Fixes cut-off LLM responses)
5. **Report**: `python3 report.py` (Generates HTML and Excel reports)

## Outputs

Reports are generated in the `outputs/` directory:

- `report_latest.html`: Visual summary for quick review.
- `report_latest.xlsx`: Detailed spreadsheet for tracking review statuses (Draft / Reviewed / Accepted / Rejected).
