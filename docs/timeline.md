# Project Timeline & Milestones

This timeline outlines the development roadmap for Synthetic Danger, moving from the current Proof-of-Concept (PoC) towards a stable, production-ready tool for robotic safety engineering.

## Current State: Phase 0 - Proof of Concept (Q1 2026)
*Done or In-Progress*

- [x] Core hazard generation pipeline (Analyze -> Normalize -> Enrich).
- [x] Basic safety ontology (Mechanical, Electrical, Software, etc.).
- [x] Automatic JSON truncation repair.
- [x] HTML & Excel report generation.
- [x] Pixi integration for automated workflows.
- [x] Structured documentation site.

## Phase 1 - Reliability & Validation (Q2 2026)
*Focus: Accuracy and Trust*

- **Stricter Input Validation**: Implement JSON schema enforcement for YAML inputs to catch configuration errors early.
- **Model Benchmarking**: Evaluate different LLM models (e.g., Gemini 1.5 Pro vs. GPT-4o) specifically for safety reasoning accuracy.
- **Reference Dataset**: Create a "gold standard" dataset of human-reviewed hazards to measure tool performance.
- **Enrich-time Truncation Recovery**: Move repair logic directly into the enrichment loop to reduce latency.

## Phase 2 - Enterprise Integration (Q3 2026)
*Focus: Workflow & Collaboration*

- **FMEA/FTA Export**: Add specific exporters for standard safety formats like Failure Mode and Effects Analysis (FMEA).
- **Reviewer Web Interface**: Transition from static HTML to an interactive web dashboard for real-time collaborative reviews.
- **Contextual Knowledge Base**: Allow users to provide custom safety manuals or industry standards (e.g., ISO 13849) as RAG context.
- **API Access**: Expose the pipeline as a REST API for integration into larger PLM (Product Lifecycle Management) systems.

## Phase 3 - V1.0 Release (Q4 2026)
*Focus: Stability & Scale*

- **Compliance Evidence Locker**: Immutable logging of all generation steps and human decisions for audit trails.
- **Multi-Agent Orchestration**: Specialized agents for different categories (e.g., a "Mechanical Specialist" vs "Cybersecurity Specialist").
- **Security Hardening**: Secure handling of proprietary system definitions and local LLM deployment options.
- **Public V1.0 stable release**.
