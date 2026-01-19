# Usage Guide

## Execution Modes

### YAML Mode (Recommended)

Edit `templates/system_input.yaml` and run:

```bash
pixi run run
```

### TXT Mode (Legacy)

If you prefer free-text:

```bash
pixi run python3 run.py --mode txt --txt data/system.txt
```

## Processing Pipeline

The full pipeline consists of several steps to ensure high-quality, normalized output:

1. **Analyze**: `pixi run analyze` (Initial hazard identification)
2. **Normalize**: `pixi run normalize` (Collapses similar components/phases)
3. **Enrich**: `pixi run enrich` (Adds requirements and verification ideas)
4. **Repair**: `pixi run repair` (Fixes cut-off LLM responses)
5. **Report**: `pixi run report` (Generates HTML and Excel reports)


## Outputs

Reports are generated in the `outputs/` directory:

- `report_latest.html`: Visual summary for quick review.
- `report_latest.xlsx`: Detailed spreadsheet for tracking review statuses (Draft / Reviewed / Accepted / Rejected).
