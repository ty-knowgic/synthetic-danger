# Synthetic Danger (PoC)

Synthetic Danger generates **draft hazards**, **safety requirements**, and **verification ideas** from structured system definitions, enabling faster robotic safety reviews with LLM support.

> [!WARNING]
> This tool generates *draft artifacts* to accelerate safety reviews. It does **not** certify compliance with any safety standard. Final validation must be performed by qualified safety engineers.

---

## Documentation

For detailed guides, architecture, and troubleshooting, please refer to our documentation:

👉 **[Go to Documentation](docs/index.md)** (Rendered via Quarto)

---

## Quick Start

### Installation

Ensure you have [Pixi](https://pixi.sh) installed.

```bash
pixi install
```

### Run

```bash
# Generate hazards
pixi run run

# Validate input
pixi run validate

# Render documentation
pixi run docs
```


---

## License

Safety Danger is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**. 
See the [LICENSE](LICENSE) file for details.
