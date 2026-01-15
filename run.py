import os
import json
import time
import re
import argparse
import sys
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from jsonschema import Draft202012Validator

from openai import OpenAI

# Optional dependency (recommended): pip install pyyaml
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# Optional: YAML input validation (validate_input.py)
try:
    from validate_input import validate_system_input  # type: ignore
except Exception:  # pragma: no cover
    validate_system_input = None  # type: ignore


# -------- Paths --------
SYSTEM_TXT_PATH = "data/system.txt"
SYSTEM_YAML_PATH = "templates/system_input.yaml"  # structured input
ONTOLOGY_PATH = "ontology/hazard_ontology.json"
# JSON Schema for hazard items (kept under schemas/ to avoid directory name drift)
SCHEMA_PATH = "schemas/hazard.schema.json"
OUTPUT_PATH = "outputs/hazards.json"


# -------- Config --------
DEFAULT_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.2

MAX_RETRIES_JSON_REPAIR = 3
MAX_RETRIES_SCHEMA_REPAIR = 2

# Start modest for quality check (you can override by env TARGET_PER_CATEGORY)
DEFAULT_TARGET_PER_CATEGORY = 15

# Strictness controls
MIN_PROPAGATION_STEPS = 4
FORBIDDEN_TASK_PHASES = {"Any", "N/A", "NA", "Unknown", ""}

# -------- Utility --------
def ensure_dirs():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def archive_outputs_dir(outputs_dir: str, archive_root: str) -> Optional[str]:
    if not os.path.isdir(outputs_dir):
        return None
    try:
        entries = os.listdir(outputs_dir)
    except FileNotFoundError:
        return None
    if not entries:
        return None
    os.makedirs(archive_root, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(archive_root, f"outputs_{timestamp}")
    shutil.move(outputs_dir, dest)
    return dest

def run_step(args: List[str]) -> None:
    print(f"[info] running: {' '.join(args)}")
    subprocess.run(args, check=True)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: str) -> Any:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Install with: pip install pyyaml\n"
            "Or set --mode txt (or SYSTEM_INPUT_MODE=txt) to use data/system.txt\n"
            "(YAML can be .yaml or .yml)"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    # allow a single string
    s = str(x).strip()
    return [s] if s else []


def _as_map(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def yaml_to_system_text(doc: Dict[str, Any]) -> str:
    """Deterministically convert structured YAML into a concise system description string."""
    ov = _as_map(doc.get("system_overview"))
    comps = _as_map(doc.get("components"))
    env = _as_map(doc.get("environment"))

    purpose = str(ov.get("purpose", "")).strip()
    mode = str(ov.get("operating_mode", "")).strip()
    goal = str(ov.get("safety_goal", "")).strip()

    # Components are expected as flat strings or small nested maps; we flatten simply.
    def fmt_components(c: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        for k in sorted(c.keys()):
            v = c[k]
            if isinstance(v, dict):
                # flatten one level
                inner = ", ".join(f"{ik}={iv}" for ik, iv in v.items())
                lines.append(f"{k}: {inner}".strip())
            elif isinstance(v, list):
                inner = ", ".join(str(i) for i in v)
                lines.append(f"{k}: {inner}".strip())
            else:
                s = str(v).strip()
                if s:
                    lines.append(f"{k}: {s}")
        return lines

    task_phases = _as_list(doc.get("tasks"))
    assumptions = _as_list(doc.get("assumptions"))

    # Allow both 'human_presence' and 'guarding' keys, but keep deterministic ordering
    env_lines: List[str] = []
    for k in sorted(env.keys()):
        v = str(env.get(k, "")).strip()
        if v:
            env_lines.append(f"{k}: {v}")

    parts: List[str] = []
    parts.append("SYSTEM OVERVIEW")
    if purpose:
        parts.append(f"- Purpose: {purpose}")
    if mode:
        parts.append(f"- Operating mode: {mode}")
    if goal:
        parts.append(f"- Safety goal: {goal}")

    parts.append("\nCOMPONENTS")
    for line in fmt_components(comps):
        parts.append(f"- {line}")

    parts.append("\nTASK PHASES")
    for t in task_phases:
        parts.append(f"- {t}")

    parts.append("\nENVIRONMENT")
    if env_lines:
        for line in env_lines:
            parts.append(f"- {line}")
    else:
        parts.append("- (not specified)")

    parts.append("\nASSUMPTIONS")
    if assumptions:
        for a in assumptions:
            parts.append(f"- {a}")
    else:
        parts.append("- (not specified)")

    return "\n".join(parts).strip() + "\n"


def load_system_input(mode: str, system_txt_path: str, system_yaml_path: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """Load system input from YAML (structured) or TXT (free text).

    Returns: (system_text, system_yaml_dict_or_none, source)
    source is one of: 'yaml' or 'txt'
    """
    """Load system input from YAML (structured) or TXT (free text).

    Returns: (system_text, system_yaml_dict_or_none, source)
    source is one of: 'yaml' or 'txt'
    """
    mode = (mode or "auto").strip().lower()

    # Resolve paths relative to current working directory for clearer debugging
    cwd = os.getcwd()
    yaml_path = os.path.abspath(system_yaml_path)
    txt_path = os.path.abspath(system_txt_path)

    yaml_exists = os.path.exists(yaml_path)
    txt_exists = os.path.exists(txt_path)

    # Prefer YAML when available unless explicitly forced to txt
    if mode in {"yaml", "auto"}:
        if yaml_exists:
            y = read_yaml(yaml_path)
            if not isinstance(y, dict):
                raise RuntimeError(f"YAML root must be a mapping/dict: {yaml_path}")
            return yaml_to_system_text(y), y, "yaml"
        if mode == "yaml":
            raise RuntimeError(
                "YAML mode was requested but the YAML file was not found.\n"
                f"- cwd:  {cwd}\n"
                f"- yaml: {yaml_path} (missing)\n\n"
                "Fix: ensure the file exists OR pass an explicit path, e.g.:\n"
                "  python3 run.py --mode yaml --yaml templates/system_input.yaml\n"
                "Or verify with:\n"
                "  ls -la templates\n"
            )

    if mode in {"txt", "auto"}:
        if txt_exists:
            return read_text(txt_path), None, "txt"
        if mode == "txt":
            raise RuntimeError(
                "TXT mode was requested but the TXT file was not found.\n"
                f"- cwd: {cwd}\n"
                f"- txt: {txt_path} (missing)\n\n"
                "Fix: ensure the file exists OR pass an explicit path, e.g.:\n"
                "  python3 run.py --mode txt --txt data/system.txt\n"
            )

    raise RuntimeError(
        "No system input found. Provide one of:\n"
        f"- YAML: {yaml_path} ({'found' if yaml_exists else 'missing'})\n"
        f"- TXT : {txt_path} ({'found' if txt_exists else 'missing'})\n"
        f"- cwd : {cwd}\n\n"
        "Tips:\n"
        "- Check filenames/case: templates/system_input.yaml vs .yml\n"
        "- Check location: run from repo root (synthetic-danger)\n"
        "- Or pass explicit paths: --yaml /path/to/file.yaml\n"
    )


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s)
        s = re.sub(r"\n```$", "", s).strip()
    return s


def try_parse_json(s: str) -> Any:
    s = strip_code_fences(s)
    return json.loads(s)


def schema_errors_for(hazards: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[str]:
    v = Draft202012Validator(schema)
    errs = []
    for i, h in enumerate(hazards):
        for e in v.iter_errors(h):
            errs.append(f"item[{i}]: {e.message}")
    return errs


def openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment/.env")
    return OpenAI(api_key=api_key)


def chat_text(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=TEMPERATURE,
    )
    return resp.output_text


# -------- Prompts --------
def build_component_extraction_prompt(system_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a senior robotics safety engineer. Extract structured facts. Output JSON only."},
        {"role": "user", "content": f"""
Extract the system components and assumptions from the description below.
Return ONLY valid JSON with this exact shape:

{{
  "sensors": [string],
  "actuators": [string],
  "compute_software": [string],
  "control_modules": [string],
  "human_interfaces": [string],
  "environment_assumptions": [string]
}}

Description:
{system_text}
""".strip()},
    ]


def build_task_extraction_prompt(system_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a senior robotics safety engineer. Extract task structure. Output JSON only."},
        {"role": "user", "content": f"""
Extract the task phases and success/failure conditions from the description below.
Return ONLY valid JSON with this exact shape:

{{
  "task_phases": [string],
  "success_condition": string,
  "notable_failure_outcomes": [string]
}}

Description:
{system_text}
""".strip()},
    ]


def build_hazard_generation_prompt(
    system_text: str,
    components: Dict[str, Any],
    task: Dict[str, Any],
    category: str,
    allowed_subtypes: List[str],
    n_target: int,
) -> List[Dict[str, str]]:
    allowed_subtypes_str = ", ".join([f'"{x}"' for x in allowed_subtypes])
    task_phases = task.get("task_phases", [])
    task_phases_str = ", ".join([f'"{p}"' for p in task_phases]) if isinstance(task_phases, list) else ""

    # This is the key: enforce deeper propagation + ban "Any" + require internal state in steps 2&3
    return [
        {"role": "system", "content": "You are a robotics functional safety engineer. You MUST output JSON only. No markdown."},
        {"role": "user", "content": f"""
Generate {n_target} distinct hazards for the category "{category}" for the robot/system described.

Hard rules (must follow):
- Output ONLY a JSON array (no markdown, no explanation).
- Each item MUST include these fields:
  hazard_id, category, subtype, trigger_condition, primary_failure, propagation_chain,
  final_impact, severity, likelihood, detectability, affected_components, task_phase, notes(optional)
- category MUST be exactly "{category}".
- subtype MUST be one of: [{allowed_subtypes_str}]
- hazard_id format MUST be: "HZ-{category[:3].upper()}-####" where #### is 4 digits (0001..).
- task_phase MUST be one of the extracted task_phases: [{task_phases_str}]
  DO NOT use "Any", "N/A", "Unknown", or placeholders.
- propagation_chain MUST have at least {MIN_PROPAGATION_STEPS} steps.
  * Step 1: immediate failure initiation
  * Steps 2 and 3: MUST explicitly describe internal system state changes (examples: "vacuum pressure drops", "tracking lost", "planner enters recovery mode", "latency spike triggers stale pose", "speed limit exceeded", "safety stop not triggered")
  * Final step: leads into the final_impact (accident/outcome)
- Each propagation step must be a short concrete state/action statement. Avoid vague phrases ("unexpected issue", "may fail", "problem occurs").
- Avoid duplicates: hazards must differ by trigger_condition AND propagation_chain content.

Use engineering realism for factory robotics. Prefer concrete triggers: glare, occlusion, conveyor speed variation, vacuum leak, dust, EMI, timing jitter, calibration drift, operator entry, E-stop misbehavior, etc.

System description:
{system_text}

Extracted components (JSON):
{json.dumps(components, ensure_ascii=False)}

Extracted task (JSON):
{json.dumps(task, ensure_ascii=False)}
""".strip()},
    ]


def build_json_repair_prompt(bad_output: str, error: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You fix invalid JSON. Output JSON only, nothing else."},
        {"role": "user", "content": f"""
The following text was intended to be JSON but failed to parse.
Fix it and return ONLY valid JSON.

Parse error:
{error}

Text:
{bad_output}
""".strip()},
    ]


def build_schema_repair_prompt(hazards: Any, schema_errors: List[str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You repair JSON items to satisfy a schema. Output JSON only."},
        {"role": "user", "content": f"""
The JSON array below must be repaired to satisfy the required Hazard schema.
Fix the items with errors, keep other items unchanged.
Return ONLY the repaired JSON array.

Important:
- propagation_chain MUST have at least {MIN_PROPAGATION_STEPS} steps.
- task_phase MUST NOT be 'Any' or placeholders.

Schema errors:
- {"\n- ".join(schema_errors[:60])}

JSON array:
{json.dumps(hazards, ensure_ascii=False)}
""".strip()},
    ]


# -------- Robust calls --------
def robust_json_call(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> Any:
    last = chat_text(client, model, messages)
    for attempt in range(MAX_RETRIES_JSON_REPAIR + 1):
        try:
            return try_parse_json(last)
        except Exception as e:
            if attempt >= MAX_RETRIES_JSON_REPAIR:
                raise RuntimeError(
                    f"JSON parse failed after repairs. Last error: {e}\nLast output (head):\n{last[:2000]}"
                )
            last = chat_text(client, model, build_json_repair_prompt(last, str(e)))
            time.sleep(0.2)
    raise RuntimeError("Unreachable")


def robust_schema_repair(client: OpenAI, model: str, hazards: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    errs = schema_errors_for(hazards, schema)
    if not errs:
        return hazards

    current = hazards
    for _ in range(MAX_RETRIES_SCHEMA_REPAIR):
        repaired = robust_json_call(client, model, build_schema_repair_prompt(current, errs))
        if not isinstance(repaired, list):
            repaired = current
        errs = schema_errors_for(repaired, schema)
        if not errs:
            return repaired
        current = repaired

    raise RuntimeError("Schema validation failed after repairs. Sample errors:\n" + "\n".join(errs[:40]))


# -------- Post checks / dedup --------
def normalize_for_dedup(h: Dict[str, Any]) -> str:
    key = " | ".join([
        h.get("category", ""),
        h.get("subtype", ""),
        h.get("task_phase", ""),
        h.get("trigger_condition", "")[:140],
        h.get("primary_failure", "")[:140],
        h.get("final_impact", "")[:140],
    ])
    return re.sub(r"\s+", " ", key).strip().lower()


def dedup_hazards(hazards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for h in hazards:
        k = normalize_for_dedup(h)
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


def enforce_local_rules(hazards: List[Dict[str, Any]], task_phases: List[str]) -> List[Dict[str, Any]]:
    """
    Extra guardrails beyond schema:
    - task_phase must not be forbidden placeholders
    - task_phase should be within extracted task_phases
    - propagation_chain length >= MIN_PROPAGATION_STEPS
    """
    allowed = set(task_phases) if isinstance(task_phases, list) else set()
    cleaned = []
    for h in hazards:
        phase = (h.get("task_phase") or "").strip()
        chain = h.get("propagation_chain") or []
        if phase in FORBIDDEN_TASK_PHASES:
            continue
        if allowed and phase not in allowed:
            continue
        if not isinstance(chain, list) or len(chain) < MIN_PROPAGATION_STEPS:
            continue
        cleaned.append(h)
    return cleaned


# -------- Main --------
def main():
    outputs_dir = os.path.abspath(os.path.dirname(OUTPUT_PATH))
    archive_root = os.path.abspath("outputs_archive")
    archived = archive_outputs_dir(outputs_dir, archive_root)
    if archived:
        print(f"[info] archived existing outputs to: {archived}")
    ensure_dirs()
    load_dotenv()

    ap = argparse.ArgumentParser(description="Generate hazards from system input (YAML or TXT)")
    ap.add_argument("--mode", choices=["auto", "yaml", "txt"], default=os.getenv("SYSTEM_INPUT_MODE", "auto"))
    ap.add_argument("--yaml", default=SYSTEM_YAML_PATH, help="Path to structured YAML system input")
    ap.add_argument("--txt", default=SYSTEM_TXT_PATH, help="Path to free-text system input")
    ap.add_argument("--validate", action="store_true", help="Validate YAML system input before calling the API")
    args = ap.parse_args()

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    system_text, system_yaml_doc, system_source = load_system_input(
        mode=args.mode,
        system_txt_path=args.txt,
        system_yaml_path=args.yaml,
    )

    # ---- Validate YAML input before calling the API (pre-flight gate) ----
    if args.validate and system_source == "yaml":
        if validate_system_input is None:
            raise RuntimeError(
                "validate_input.py is not importable. Ensure validate_input.py exists in repo root.\n"
                "Run from the repo root directory, or check your PYTHONPATH."
            )
        if system_yaml_doc is None or not isinstance(system_yaml_doc, dict):
            raise RuntimeError("YAML source selected but parsed YAML doc is missing/invalid.")

        errors, warnings = validate_system_input(system_yaml_doc)

        if warnings:
            print("[warn] YAML input warnings:")
            for w in warnings:
                print(f"  - {w}")

        if errors:
            print("[err] YAML input errors:")
            for e in errors:
                print(f"  - {e}")
            print("[err] Aborting before OpenAI API calls due to invalid input.")
            sys.exit(1)
        print(f"[info] model={model}")
        print(f"[info] system_input_source={system_source}")
        print(f"[info] system_input_path={args.yaml if system_source == 'yaml' else args.txt}")

    target_per_category = int(os.getenv("TARGET_PER_CATEGORY", str(DEFAULT_TARGET_PER_CATEGORY)))

    ontology = load_json(ONTOLOGY_PATH)
    schema = load_json(SCHEMA_PATH)

    client = openai_client()

    print("[info] extracting components...")
    components = robust_json_call(client, model, build_component_extraction_prompt(system_text))
    print("[info] extracting task structure...")
    task = robust_json_call(client, model, build_task_extraction_prompt(system_text))

    if not isinstance(components, dict) or not isinstance(task, dict):
        raise RuntimeError("Extraction failed: components/task is not dict JSON")

    task_phases = task.get("task_phases", [])
    if not isinstance(task_phases, list) or not task_phases:
        raise RuntimeError("Task extraction failed: task_phases missing or not list")

    categories: Dict[str, List[str]] = ontology.get("categories", {})
    if not categories:
        raise RuntimeError("Ontology categories missing")

    all_hazards: List[Dict[str, Any]] = []

    for cat, subtypes in categories.items():
        print(f"[info] generating hazards for category={cat} target={target_per_category} ...")
        msgs = build_hazard_generation_prompt(
            system_text=system_text,
            components=components,
            task=task,
            category=cat,
            allowed_subtypes=subtypes,
            n_target=target_per_category,
        )

        hazards = robust_json_call(client, model, msgs)
        if not isinstance(hazards, list):
            raise RuntimeError(f"Expected list of hazards for {cat}, got {type(hazards)}")

        # Schema repair
        hazards = robust_schema_repair(client, model, hazards, schema)

        # Extra local filters (remove Any, wrong phase, short chain)
        hazards = enforce_local_rules(hazards, task_phases)

        all_hazards.extend(hazards)
        time.sleep(0.25)

    print(f"[info] total hazards before dedup: {len(all_hazards)}")
    all_hazards = dedup_hazards(all_hazards)
    print(f"[info] total hazards after dedup:  {len(all_hazards)}")

    out = {
        "system_input": system_text,
        "system_input_source": system_source,
        "system_input_yaml": system_yaml_doc,  # None when TXT is used
        "components": components,
        "task": task,
        "hazards": all_hazards,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote: {OUTPUT_PATH}")

    if not all_hazards:
        print("[warn] no hazards generated; skipping enrichment pipeline.")
        run_step([sys.executable, "report.py", "--input", OUTPUT_PATH, "--outdir", "outputs", "--basename", "report"])
        return

    run_step([sys.executable, "normalize.py"])
    run_step([sys.executable, "enrich.py", "--resume"])
    run_step([sys.executable, "repair_truncation.py"])
    run_step(
        [
            sys.executable,
            "report.py",
            "--input",
            "outputs/hazards_enriched_repaired.json",
            "--outdir",
            "outputs",
            "--basename",
            "report",
        ]
    )


if __name__ == "__main__":
    main()
