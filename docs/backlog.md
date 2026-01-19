# Project Backlog

This backlog tracks planned features, technical debt, and ideas for Synthetic Danger. Items are categorized by their nature and are prioritized based on the [Project Timeline](timeline.md).

## High Priority (Active Development)

- **[Feature] Stricter YAML Validation**: Enforce JSON schema at the input stage (`validate_input.py`) to prevent LLM hallucination due to malformed system definitions.
- **[Feature] Truncation Recovery in Loop**: Integrate `repair_truncation.py` logic directly into the `enrich.py` loop to fix issues as they occur.
- **[Refactor] Template-based Reporting**: Replace manual HTML string concatenation in `report.py` with a template engine like **Jinja2** for better maintainability and theme support.

## Medium Priority (Enhancements)

- **[Feature] Multi-Model Comparison**: Add a script to run the same input through multiple models (e.g., GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet) and compare hazard count/quality.
- **[Docs] Example Cookbook**: Create a library of example YAML inputs for common robotic systems (Autonomous Mobile Robots, 6-DOF Cobots, Drones).
- **[Testing] Unit Test Suite**: Add tests for core logic, especially normalization and repairing truncation.
- **[Feature] Recursive Enrichment**: Allow the tool to drill down into high-risk hazards to generate even more specific "Level 3" requirements.

## Low Priority (Future Ideas)

- **[Integration] JIRA/ADO Exporter**: Direct sync of identified hazards into engineering ticket systems.
- **[Integration] CI/CD Documentation**: Automatically render and publish this documentation site on every push to `main`.
- **[UI] Interactive Dashboard**: A lightweight React/Next.js frontend to replace the static HTML report for more complex review sessions.
- **[Feature] Local LLM Support**: Support for running the pipeline against local models (via Ollama or vLLM) for high-confidentiality system reviews.

## Technical Debt

- **Type Hinting**: Complete Python type hinting across all modules (`run.py`, `enrich.py`, etc.).
- **Dependency Cleanup**: Review `pyproject.toml` to ensure only strictly necessary packages are included in the base environment.
- **CLI Consistency**: Standardize argument names across all scripts (e.g., always using `--input` vs `--file`).
