import json
import re
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Set

# Optional: better fuzzy matching
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


IN_PATH = "outputs/hazards.json"
OUT_PATH = "outputs/hazards_normalized.json"
REPORT_PATH = "outputs/normalize_report.txt"

# ----------------------------
# Config: normalization policy
# ----------------------------

# Lowercase + strip punctuation for matching
def norm_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9\-\s/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# A few common aliases you’ll almost certainly see.
# Add aggressively as you observe outputs.
DEFAULT_ALIAS_MAP = {
    # ROS / software stack
    "ros 2": "ROS2",
    "ros2": "ROS2",
    "ros2 compute_software": "ROS2",
    "ros2 middleware": "ROS2",
    "ros2 node": "ROS2",
    "motion planning module": "Motion planner",
    "motion planner": "Motion planner",
    "planner": "Motion planner",
    "grasp pose estimation module": "Grasp pose estimator",
    "grasp pose estimator": "Grasp pose estimator",
    "pose estimator": "Grasp pose estimator",

    # Sensors / perception
    "rgb camera": "RGB camera",
    "camera": "RGB camera",
    "overhead camera": "RGB camera",
    "fixed led lighting": "LED lighting",
    "led lighting": "LED lighting",
    "lighting": "LED lighting",

    # Actuators / mechanics
    "vacuum suction gripper": "Vacuum gripper",
    "vacuum gripper": "Vacuum gripper",
    "suction gripper": "Vacuum gripper",
    "robotic arm": "Robot arm",
    "6 axis industrial robotic arm": "Robot arm",
    "6-axis industrial robotic arm": "Robot arm",
    "robot": "Robot arm",

    # HRI / safety
    "e stop": "E-Stop",
    "e-stop": "E-Stop",
    "emergency stop": "E-Stop",
    "emergency stop button": "E-Stop",
    "emergency stop (e-stop) button": "E-Stop",
    "operator": "Human operator",
    "human operator": "Human operator",
    "operators": "Human operator",
    "human interfaces": "E-Stop",  # force away from useless abstraction

    # Environment / peripherals
    "conveyor": "Conveyor",
    "moving conveyor belt": "Conveyor",
    "conveyor belt": "Conveyor",
    "tray": "Target tray",
    "target tray": "Target tray",
    "factory environment": "Factory environment",

        # Power / electrical
    "power supply": "Power supply",
    "power supply wiring": "Power supply",
    "power supply fuse": "Power supply",

    # Control electronics
    "motor controller": "Motor controller",
    "control modules": "Control system",

    # Robot internal sensing
    "joint encoders": "Joint encoders",

    # Vacuum sensing (promote to explicit component)
    "vacuum pressure sensor": "Vacuum pressure sensor",

    # Conveyor sensing
    "conveyor system": "Conveyor",
    "conveyor belt sensors": "Conveyor sensors",

    "emergency stop (e-stop) button": "E-Stop",
    "emergency stop (e-stop) button.": "E-Stop"
}

# If you want to forbid vague component names completely, list them:
FORBIDDEN_COMPONENTS = {
    "human_interfaces",
    "compute_software",
    "control_modules",
    "sensors",
    "actuators",
    "environment_assumptions",
    "system",
    "subsystem",
    "component",
}

# ----------------------------
# Extract canonical components
# ----------------------------

def extract_canonical_components(data: Dict[str, Any]) -> List[str]:
    """
    Build a canonical component list from extracted components + obvious nouns.
    The goal is to have a stable reference set for affected_components normalization.
    """
    comps = data.get("components", {}) or {}
    canon: Set[str] = set()

    def add_list(k: str):
        v = comps.get(k, [])
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    canon.add(x.strip())

    for k in ["sensors", "actuators", "compute_software", "control_modules", "human_interfaces"]:
        add_list(k)

    # Add some likely peripherals mentioned in system_input
    sys_text = data.get("system_input", "") or ""
    # naive keyword additions (you can expand)
    keywords = [
        ("conveyor", "Conveyor"),
        ("tray", "Target tray"),
        ("e-stop", "E-Stop"),
        ("emergency stop", "E-Stop"),
        ("ros2", "ROS2"),
        ("rgb camera", "RGB camera"),
        ("led", "LED lighting"),
        ("vacuum", "Vacuum gripper"),
        ("robotic arm", "Robot arm"),
    ]
    low = sys_text.lower()
    for k, v in keywords:
        if k in low:
            canon.add(v)

    # Normalize final canonical: collapse to nice title-ish casing when possible via alias map
    # We will keep raw items too, but canonical must be stable.
    nice = set()
    for c in canon:
        ck = norm_key(c)
        if ck in DEFAULT_ALIAS_MAP:
            nice.add(DEFAULT_ALIAS_MAP[ck])
        else:
            # Heuristic: preserve original but strip redundant underscores
            nice.add(c.replace("_", " ").strip())
    return sorted(nice)


def build_alias_map(canonical: List[str]) -> Dict[str, str]:
    """
    Seed alias map with DEFAULT_ALIAS_MAP plus identity mappings for canonical items.
    Keys are normalized.
    """
    amap = {}
    for k, v in DEFAULT_ALIAS_MAP.items():
        amap[norm_key(k)] = v
    for c in canonical:
        amap[norm_key(c)] = c
    return amap


# ----------------------------
# Matching logic
# ----------------------------

def best_match_component(s: str, canonical: List[str], alias_map: Dict[str, str]) -> Tuple[str, str]:
    """
    Returns (normalized_component, reason)
    reason: 'alias' | 'exact' | 'fuzzy' | 'raw'
    """
    if not s or not isinstance(s, str):
        return ("", "raw")
    raw = s.strip()
    k = norm_key(raw)

    if k in alias_map:
        return (alias_map[k], "alias")

    # exact normalized match against canonical
    canon_keys = {norm_key(c): c for c in canonical}
    if k in canon_keys:
        return (canon_keys[k], "exact")

    # fuzzy match
    if HAS_RAPIDFUZZ and canonical:
        choices = canonical
        # match on normalized string to avoid punctuation weirdness
        result = process.extractOne(raw, choices, scorer=fuzz.WRatio)
        if result and result[1] >= 88:  # threshold
            return (result[0], "fuzzy")

    # fallback: keep raw but cleaned
    return (raw, "raw")


def normalize_task_phase(phase: str, allowed: List[str]) -> Tuple[str, str]:
    """
    Map phase to one of allowed phases. Returns (phase, reason).
    """
    if not phase or not isinstance(phase, str):
        return ("", "raw")
    raw = phase.strip()

    # exact
    if raw in allowed:
        return (raw, "exact")

    # case-insensitive exact
    low_allowed = {a.lower(): a for a in allowed}
    if raw.lower() in low_allowed:
        return (low_allowed[raw.lower()], "casefold")

    # fuzzy
    if HAS_RAPIDFUZZ and allowed:
        result = process.extractOne(raw, allowed, scorer=fuzz.WRatio)
        if result and result[1] >= 88:
            return (result[0], "fuzzy")

    return (raw, "raw")


# ----------------------------
# Main normalization
# ----------------------------

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Not found: {IN_PATH}")

    with open(IN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    hazards: List[Dict[str, Any]] = data.get("hazards", []) or []
    task = data.get("task", {}) or {}
    allowed_phases = task.get("task_phases", []) or []
    if not isinstance(allowed_phases, list):
        allowed_phases = []

    canonical = extract_canonical_components(data)
    alias_map = build_alias_map(canonical)

    comp_reason_counter = Counter()
    phase_reason_counter = Counter()
    unknown_components = Counter()
    forbidden_components_hits = 0

    normalized_hazards = []
    for h in hazards:
        h2 = dict(h)

        # Normalize task_phase
        phase = h.get("task_phase", "")
        new_phase, preason = normalize_task_phase(phase, allowed_phases)
        phase_reason_counter[preason] += 1
        h2["task_phase"] = new_phase

        # Normalize affected_components
        comps = h.get("affected_components", [])
        new_comps = []
        if isinstance(comps, list):
            for c in comps:
                if not isinstance(c, str):
                    continue
                if norm_key(c) in FORBIDDEN_COMPONENTS:
                    forbidden_components_hits += 1
                    continue
                mapped, reason = best_match_component(c, canonical, alias_map)
                comp_reason_counter[reason] += 1
                if not mapped:
                    continue
                # Track unknowns
                if reason == "raw":
                    unknown_components[mapped] += 1
                new_comps.append(mapped)

        # de-dup within one hazard
        seen = set()
        deduped = []
        for c in new_comps:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        h2["affected_components"] = deduped
        normalized_hazards.append(h2)

    # Overall unknown ratio
    total_comp_mentions = sum(comp_reason_counter.values())
    raw_mentions = comp_reason_counter["raw"]
    raw_ratio = (raw_mentions / total_comp_mentions) if total_comp_mentions else 0.0

    # Task phase not in allowed (raw)
    raw_phase = phase_reason_counter["raw"]
    raw_phase_ratio = (raw_phase / len(hazards)) if hazards else 0.0

    # Save normalized file
    out = dict(data)
    out["normalization"] = {
        "canonical_components": canonical,
        "component_match_stats": dict(comp_reason_counter),
        "task_phase_match_stats": dict(phase_reason_counter),
        "raw_component_ratio": raw_ratio,
        "raw_task_phase_ratio": raw_phase_ratio,
        "forbidden_components_removed": forbidden_components_hits,
        "notes": "Raw components indicate missing alias mappings or canonical list gaps. Expand DEFAULT_ALIAS_MAP as needed."
    }
    out["hazards"] = normalized_hazards

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Produce report
    lines = []
    lines.append("=== NORMALIZATION REPORT ===")
    lines.append(f"Input:  {IN_PATH}")
    lines.append(f"Output: {OUT_PATH}")
    lines.append("")
    lines.append(f"Hazards: {len(hazards)}")
    lines.append("")
    lines.append("Canonical components (from extraction + heuristics):")
    for c in canonical:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append("Component match reasons:")
    for k, v in comp_reason_counter.most_common():
        lines.append(f"  {k}: {v}")
    lines.append(f"Raw component ratio: {raw_ratio:.2%}")
    lines.append(f"Forbidden components removed: {forbidden_components_hits}")
    lines.append("")
    lines.append("Task phase match reasons:")
    for k, v in phase_reason_counter.most_common():
        lines.append(f"  {k}: {v}")
    lines.append(f"Raw task_phase ratio: {raw_phase_ratio:.2%}")
    lines.append("")
    lines.append("Top unknown (raw) affected_components (candidates for alias map):")
    for k, v in unknown_components.most_common(25):
        lines.append(f"  {k}: {v}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[ok] wrote normalized hazards: {OUT_PATH}")
    print(f"[ok] wrote report:            {REPORT_PATH}")
    print(f"[info] raw component ratio:   {raw_ratio:.2%}")
    print(f"[info] raw task_phase ratio:  {raw_phase_ratio:.2%}")
    if not HAS_RAPIDFUZZ:
        print("[warn] rapidfuzz not installed; fuzzy matching is disabled. (pip install rapidfuzz)")

if __name__ == "__main__":
    main()