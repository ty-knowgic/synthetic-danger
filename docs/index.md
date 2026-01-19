# Synthetic Danger (PoC)

Synthetic Danger generates **draft hazards**, **safety requirements**, and **verification ideas** from structured system definitions, enabling faster robotic safety reviews with LLM support.

> [!WARNING]
> This tool generates *draft artifacts* to accelerate safety reviews. It does **not** certify compliance with any safety standard (e.g., ISO 10218, ISO 13849, IEC 61508). Final validation and approval must be performed by qualified safety engineers.

## Quick Start

### 1. Installation

Ensure you have [Pixi](https://pixi.sh) installed, then run:

```bash
pixi install
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=YOUR_API_KEY
```

### 3. Run a sample

```bash
pixi run run
```


The output will be generated in `outputs/hazards.json`.
