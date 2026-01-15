# FAQ

## For Users (run and consume outputs)

Q1. What does this tool do?
A1. It generates draft hazards/requirements/verification ideas from structured input and outputs HTML/XLSX reports.

Q2. What is the intended use?
A2. The outputs are draft artifacts for human review, not safety guarantees or compliance evidence.

Q3. Where should I put the input?
A3. For YAML mode, use `templates/system_input.yaml` or pass `--yaml <path>`.

Q4. Can I still use TXT input?
A4. Yes. Use `--mode txt --txt <path>`.

Q5. Can I validate YAML input?
A5. Run `python3 validate_input.py --yaml <path>` or use `run.py --validate`.

Q6. Where are outputs generated?
A6. By default, under `./outputs`. `report.html` and `report.xlsx` are written there too.

Q7. Will existing outputs be overwritten?
A7. If `outputs` is non-empty, it is archived to `outputs_archive/outputs_<timestamp>` before the run.

Q8. Where does the "System Input" in report.html come from?
A8. It comes from the YAML/TXT input, stored as `system_input` in the run output.

Q9. Does the Generated timestamp update each run?
A9. Yes, it is set at report generation time.

Q10. Will report.xlsx always update?
A10. As long as hazards are generated and `report.py` runs, it will be updated.

Q11. What happens if hazards are empty?
A11. The report is still generated, and Summary shows "No hazards generated".

Q12. How can I improve output quality?
A12. Make `task_phases` and `components` more specific and avoid vague wording.

Q13. What dependencies are required?
A13. `openai`, `python-dotenv`, `jsonschema`, `pyyaml` are required; `pandas` and `openpyxl` are needed for XLSX.

Q14. What if the run is slow or stalls?
A14. OpenAI API latency is the main factor; lowering `TARGET_PER_CATEGORY` helps.

Q15. What is a recommended review flow?
A15. Use HTML to scan, then use XLSX to fill review fields and make decisions.

## For Modifiers (change the codebase)

Q1. Where does the pipeline start?
A1. The entry point is `run.py`, which loads input and generates hazards.

Q2. Where is hazard generation implemented?
A2. In `run.py` via `build_hazard_generation_prompt` and `robust_json_call`.

Q3. What is the data flow from YAML to report?
A3. `run.py` writes `outputs/hazards.json`, then `normalize.py` -> `enrich.py` -> `repair_truncation.py` -> `report.py`.

Q4. Where is report generation implemented?
A4. `report.py` reads `hazards_enriched_repaired.json` and writes HTML/XLSX.

Q5. What does report.html use for System Input?
A5. It displays `system_input` from the input JSON.

Q6. How do I avoid crashes on empty hazards?
A6. Treat `hazards` as a list and handle the empty case in `report.py`.

Q7. How do I change the output directory?
A7. Update both `report.py --outdir` usage and the call site in `run.py`.

Q8. Where is the outputs archive logic?
A8. In `run.py` as `archive_outputs_dir`, moving to `outputs_archive`.

Q9. Why normalize?
A9. It reduces component/task_phase variability to stabilize downstream quality.

Q10. Is re-running enrich safe?
A10. Use `--resume` to keep existing enrichment and fill only missing hazards.

Q11. What triggers repair_truncation?
A11. It regenerates `safety_requirement` / `verification_idea` containing "..." or "…".

Q12. How do I switch models?
A12. Use `OPENAI_MODEL` or `MODEL`, or pass `enrich.py --model`.

Q13. How do I modify the hazard schema?
A13. Edit `schemas/hazard.schema.json` and ensure schema repair still passes.

Q14. How do I change report columns or order?
A14. Edit `to_rows` and the column lists in `report.py`.

Q15. What is the minimum safe rule for new features?
A15. Keep CLI args stable and preserve the default `outputs` location.
