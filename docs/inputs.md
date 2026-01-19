# Inputs and Ontology

## Input Schema

The recommended input format is YAML. You can find the schema and examples in the `schemas/` and `templates/` folders.

### System Definition

The YAML input should define:
- `system_name`: Name of the system under review.
- `description`: A high-level description of functionality.
- `components`: Key hardware/software modules.
- `task_phases`: Specific states or operations (e.g., Startup, Maintenance, Emergency Stop).

## Ontology

Synthetic Danger uses a structured ontology to categorize hazards. This ensures consistency across different reviews.

- **Hazard Categories**: Based on safety standards like ISO 12100.
- **Requirement Levels**: Draft requirements are generated to map directly to identified hazardous situations.
