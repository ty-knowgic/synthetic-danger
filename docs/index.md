# Synthetic Danger (PoC)

Synthetic Danger generates **draft hazards**, **safety requirements**, and **verification ideas** from structured system definitions, enabling faster robotic safety reviews with LLM support.

> [!WARNING]
> This tool generates *draft artifacts* to accelerate safety reviews. It does **not** certify compliance with any safety standard (e.g., ISO 10218, ISO 13849, IEC 61508). Final validation and approval must be performed by qualified safety engineers.

## Quick Start

### 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install openai python-dotenv jsonschema pandas openpyxl pyyaml
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=YOUR_API_KEY
```

### 3. Run a sample

```bash
python3 run.py --mode yaml --yaml templates/system_input.yaml --validate
```

The output will be generated in `outputs/hazards.json`.
