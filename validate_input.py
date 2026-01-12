import os
import sys
import argparse
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# Optional dependency (recommended): pip install pyyaml
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


DEFAULT_YAML_PATH = "templates/system_input.yaml"


def read_yaml(path: str) -> Any:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Install with: pip install pyyaml\n"
            "(YAML can be .yaml or .yml)"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def err(msg: str) -> str:
    return msg


def warn(msg: str) -> str:
    return msg


def validate_system_input(doc: Any) -> Tuple[List[str], List[str]]:
    """
    Returns: (errors, warnings)
    Errors should block execution.
    Warnings are advisory but allow execution.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(doc, dict):
        return [err("YAML root must be a mapping/dict (key-value object).")], []

    # ----- system_overview -----
    ov = doc.get("system_overview")
    if not isinstance(ov, dict):
        errors.append(err("Missing or invalid 'system_overview' (must be a mapping)."))
        ov = {}

    purpose = ov.get("purpose")
    operating_mode = ov.get("operating_mode")
    safety_goal = ov.get("safety_goal")

    if not is_nonempty_str(purpose):
        errors.append(err("system_overview.purpose is required and must be a non-empty string."))

    if operating_mode is not None and not is_nonempty_str(operating_mode):
        errors.append(err("system_overview.operating_mode must be a non-empty string if provided."))

    if safety_goal is not None and not is_nonempty_str(safety_goal):
        errors.append(err("system_overview.safety_goal must be a non-empty string if provided."))

    # Optional: gentle constraints on operating_mode
    if is_nonempty_str(operating_mode):
        allowed_modes = {"manual", "semi-auto", "auto", "autonomous"}
        if operating_mode.strip().lower() not in allowed_modes:
            warnings.append(
                warn(
                    f"system_overview.operating_mode='{operating_mode}' is not in recommended set "
                    f"{sorted(allowed_modes)}. (This is a warning, not an error.)"
                )
            )

    # ----- components -----
    comps = doc.get("components")
    if not isinstance(comps, dict):
        errors.append(err("Missing or invalid 'components' (must be a mapping)."))
        comps = {}

    # Your minimal template uses these keys; treat as recommended, not mandatory
    recommended_component_keys = ["robot", "end_effector", "perception", "control"]
    missing_recommended = [k for k in recommended_component_keys if not is_nonempty_str(comps.get(k))]
    if missing_recommended:
        warnings.append(
            warn(f"components is missing recommended keys or values: {missing_recommended} (warning).")
        )

    # Ensure component values are strings (we allow dict/list too, but warn)
    for k, v in comps.items():
        if isinstance(v, (dict, list)):
            # allowed for future expansion, but warn for now
            warnings.append(warn(f"components.{k} is {type(v).__name__}; currently best as a string (warning)."))
        elif v is None or is_nonempty_str(v):
            pass
        else:
            errors.append(err(f"components.{k} must be a string (or a dict/list for advanced usage)."))

    # ----- tasks -----
    tasks = doc.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append(err("Missing or invalid 'tasks' (must be a non-empty list)."))
        tasks = []

    bad_tasks = []
    normalized_tasks: List[str] = []
    for i, t in enumerate(tasks):
        if not is_nonempty_str(t):
            bad_tasks.append(i)
        else:
            normalized_tasks.append(t.strip())

    if bad_tasks:
        errors.append(err(f"tasks entries must be non-empty strings. Invalid indices: {bad_tasks}"))

    # Duplicate tasks can cause weird distributions
    if len(set(normalized_tasks)) != len(normalized_tasks):
        warnings.append(warn("tasks contains duplicates (warning). Consider de-duplicating for stability."))

    # Banned placeholders
    forbidden = {"any", "n/a", "na", "unknown", ""}
    hit_forbidden = [t for t in normalized_tasks if t.strip().lower() in forbidden]
    if hit_forbidden:
        errors.append(err(f"tasks contains forbidden placeholder(s): {hit_forbidden}"))

    # ----- environment -----
    env = doc.get("environment")
    if env is None:
        warnings.append(warn("environment is not provided (warning)."))
        env = {}
    if env is not None and not isinstance(env, dict):
        errors.append(err("environment must be a mapping if provided."))

    if isinstance(env, dict):
        for k in ["human_presence", "guarding"]:
            if k in env and env[k] is not None and not is_nonempty_str(env[k]):
                errors.append(err(f"environment.{k} must be a non-empty string if provided."))

    # ----- assumptions -----
    assumptions = doc.get("assumptions")
    if assumptions is None:
        warnings.append(warn("assumptions not provided (warning)."))
    elif not isinstance(assumptions, list):
        errors.append(err("assumptions must be a list of strings if provided."))
    else:
        bad_a = [i for i, a in enumerate(assumptions) if not is_nonempty_str(a)]
        if bad_a:
            errors.append(err(f"assumptions entries must be non-empty strings. Invalid indices: {bad_a}"))

    # ----- unknown top-level keys (helpful for typos) -----
    known_top = {"system_overview", "components", "tasks", "environment", "assumptions"}
    extra = [k for k in doc.keys() if k not in known_top]
    if extra:
        warnings.append(warn(f"Unknown top-level keys present: {extra} (warning). Possible typos?"))

    return errors, warnings


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Validate YAML system input for synthetic-danger")
    ap.add_argument("--yaml", default=os.getenv("SYSTEM_YAML_PATH", DEFAULT_YAML_PATH))
    args = ap.parse_args()

    path = args.yaml
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        print(f"[err] YAML not found: {abs_path}")
        sys.exit(2)

    doc = read_yaml(abs_path)
    errors, warnings = validate_system_input(doc)

    print("=== VALIDATE INPUT ===")
    print(f"yaml: {abs_path}")

    if warnings:
        print("\n[warn] warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n[err] errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\n[ok] input is valid.")
    sys.exit(0)


if __name__ == "__main__":
    main()