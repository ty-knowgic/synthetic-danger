import os
import re
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# OpenAI SDK
from openai import OpenAI

# Optional: JSON Schema validation
try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None  # type: ignore


DEFAULT_INPUT = "outputs/hazards_enriched_repaired.json"
DEFAULT_OUTPUT = "outputs/hazards_reasoned.json"
DEFAULT_SCHEMA = "schemas/reason_schema.json"


def coerce_hazards_container(data: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """
    Accepts either:
      - a JSON list of hazard dicts
      - a JSON dict wrapper that contains a list of hazard dicts under a known key

    Returns:
      (container_dict_or_none, hazards_list, hazards_key_or_none)

    If container_dict_or_none is not None, callers should write back into that dict
    under hazards_key_or_none when saving.
    """
    if isinstance(data, list):
        # raw list
        return None, data, None

    if isinstance(data, dict):
        # common wrappers
        for k in ["hazards", "items", "data", "records", "results"]:
            v = data.get(k)
            if isinstance(v, list):
                return data, v, k

        # fallback: if dict looks like a single hazard, wrap it
        if "hazard_id" in data and "category" in data:
            return None, [data], None

        raise RuntimeError(
            "Input JSON is a dict but does not contain a hazards list under any of: "
            "hazards, items, data, records, results. Keys present: "
            + ", ".join(list(data.keys())[:25])
        )

    raise RuntimeError(f"Unsupported input JSON type: {type(data)}")


def write_output(path: str, container: Optional[Dict[str, Any]], hazards: List[Dict[str, Any]], hazards_key: Optional[str]) -> None:
    """Write either a raw hazards list, or a wrapper dict with hazards updated."""
    if container is None:
        save_json(path, hazards)
    else:
        if not hazards_key:
            # should not happen, but be safe
            hazards_key = "hazards"
        container[hazards_key] = hazards
        save_json(path, container)


def now_iso() -> str:
    # Use timezone-aware UTC timestamp (Python 3.12+ friendly)
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from a string.
    Works even if the model adds extra text.
    """
    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first {...} block
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in model output.")
    # A simple brace-matching scan
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception as e:
                    raise ValueError(f"Failed to parse extracted JSON object: {e}") from e
    raise ValueError("Unbalanced braces; could not extract complete JSON object.")


def render_hazard_brief(h: Dict[str, Any]) -> str:
    # Keep it short to avoid token bloat
    return json.dumps(
        {
            "hazard_id": h.get("hazard_id"),
            "category": h.get("category"),
            "subtype": h.get("subtype"),
            "task_phase": h.get("task_phase"),
            "trigger_condition": h.get("trigger_condition"),
            "primary_failure": h.get("primary_failure"),
            "propagation_chain": h.get("propagation_chain"),
            "final_impact": h.get("final_impact"),
            "severity": h.get("severity"),
            "likelihood": h.get("likelihood"),
            "detectability": h.get("detectability"),
            "risk_band": h.get("risk_band"),
            "affected_components": h.get("affected_components"),
            "observables": h.get("observables"),
            "mitigations": h.get("mitigations"),
            "safety_requirement": h.get("safety_requirement"),
            "verification_idea": h.get("verification_idea"),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_prompt(hazard_brief: str, schema: Dict[str, Any]) -> str:
    # Hard constraints to avoid SysML/formal verification rabbit hole
    return f"""
You are a safety reasoning assistant for robotics safety reviews.
Your job: generate a compact, review-friendly reasoning block for ONE hazard.

IMPORTANT CONSTRAINTS:
- Do NOT mention SysML, UML, formal verification, model checking, theorem proving.
- Do NOT invent numeric thresholds. If parameters are unknown, put them into review_aids.open_questions.
- Keep content compact: short sentences, no long paragraphs.
- Organize risk reduction controls into ISO 12100-style A/B/C buckets:
  A = inherently safe design controls
  B = protective measures (detect/stop, safeguarding)
  C = information for use (maintenance, inspection, training)

Return ONLY a single JSON object that matches this JSON Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Hazard (input):
{hazard_brief}
""".strip()


def validate_against_schema(obj: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if jsonschema is None:
        return True, None
    try:
        jsonschema.validate(instance=obj, schema=schema)  # type: ignore
        return True, None
    except Exception as e:
        return False, str(e)


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.2,
) -> str:
    """
    Compatible pattern without relying on response_format
    (since SDK versions differ).
    """
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )
    # The OpenAI SDK returns output text via output_text helper
    return resp.output_text


def should_reason(h: Dict[str, Any]) -> bool:
    # risk_band があるならそれを優先
    rb = h.get("risk_band")
    if rb in ("High", "Critical"):
        return True
    if rb is not None:
        return False

    # risk_band が無い場合は severity で代替（最短で回すための現実解）
    sev = h.get("severity")
    return sev in ("High", "Critical")


def already_reasoned(h: Dict[str, Any]) -> bool:
    # If any of the three keys exist, assume done
    return isinstance(h.get("safety_context"), dict) and isinstance(h.get("risk_story"), dict) and isinstance(h.get("review_aids"), dict)


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Generate safety reasoning for High risk_band hazards only.")
    ap.add_argument("--input", default=os.getenv("REASON_INPUT", DEFAULT_INPUT))
    ap.add_argument("--output", default=os.getenv("REASON_OUTPUT", DEFAULT_OUTPUT))
    ap.add_argument("--schema", default=os.getenv("REASON_SCHEMA", DEFAULT_SCHEMA))
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    ap.add_argument("--limit", type=int, default=0, help="Process up to N hazards (0 = no limit)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output file if present")
    ap.add_argument("--dry_run", action="store_true", help="Do not call the API; show what would be processed")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or export it in your shell.")

    schema = load_json(args.schema)
    input_data = load_json(args.input)

    input_container, input_hazards, input_hazards_key = coerce_hazards_container(input_data)

    # Resume strategy:
    # - If output exists and --resume is set, load it and use it as the base,
    #   otherwise start from input.
    base: List[Dict[str, Any]]
    output_container: Optional[Dict[str, Any]] = None
    output_hazards_key: Optional[str] = None

    if args.resume and os.path.exists(args.output):
        out_data = load_json(args.output)
        output_container, base, output_hazards_key = coerce_hazards_container(out_data)
        print(f"[info] resume=yes output_loaded={args.output} hazards={len(base)}")
    else:
        # Use input hazards; if input was wrapped, keep the wrapper so we can preserve metadata
        output_container = input_container
        output_hazards_key = input_hazards_key
        base = input_hazards
        print(f"[info] resume=no input_loaded={args.input} hazards={len(base)}")

    client = OpenAI(api_key=api_key)

    # Build target list
    targets: List[int] = []
    for idx, h in enumerate(base):
        if not isinstance(h, dict):
            continue
        if should_reason(h) and not already_reasoned(h):
            targets.append(idx)

    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    print("=== REASON PLAN ===")
    print(f"input:   {args.input}")
    print(f"output:  {args.output}")
    print(f"schema:  {args.schema}")
    print(f"model:   {args.model}")
    print(f"hazards_total: {len(base)}")
    print(f"targets(High & not yet reasoned): {len(targets)}")
    print("[info] risk_band missing count:",
          sum(1 for h in base if isinstance(h, dict) and h.get("risk_band") is None))
    print("[info] severity counts:",
          __import__("collections").Counter([h.get("severity","<missing>") for h in base if isinstance(h, dict)]))
    if len(targets) > 0:
        sample_ids = [base[i].get("hazard_id") for i in targets[:10]]
        print(f"target_ids(sample up to 10): {', '.join([x for x in sample_ids if x])}")

    if args.dry_run:
        print("[done] dry_run=yes (no API calls)")
        return

    processed = 0
    enriched = 0
    failed = 0

    # Retry settings
    max_attempts = 4
    sleep_base = 1.2

    for idx in targets:
        h = base[idx]
        hid = h.get("hazard_id", f"idx-{idx}")

        hazard_brief = render_hazard_brief(h)
        prompt = build_prompt(hazard_brief, schema)

        last_err: Optional[str] = None
        for attempt in range(1, max_attempts + 1):
            try:
                text = call_model(client, args.model, prompt, temperature=0.2)
                obj = extract_json_object(text)

                ok, msg = validate_against_schema(obj, schema)
                if not ok:
                    raise ValueError(f"Schema validation failed: {msg}")

                # Merge results
                h["safety_context"] = obj["safety_context"]
                h["risk_story"] = obj["risk_story"]
                h["review_aids"] = obj["review_aids"]
                h["reasoning_meta"] = {
                    "reasoned_at": now_iso(),
                    "reasoning_version": "v1_high_only",
                    "model": args.model,
                }

                enriched += 1
                print(f"[ok] reasoned {hid}")
                break

            except KeyboardInterrupt:
                print("\n[warn] interrupted by user (KeyboardInterrupt). Saving checkpoint...")
                write_output(args.output, output_container, base, output_hazards_key)
                raise
            except Exception as e:
                last_err = str(e)
                wait = sleep_base * (2 ** (attempt - 1))
                print(f"[err] failed {hid} attempt={attempt}/{max_attempts}: {last_err}")
                time.sleep(wait)

        processed += 1

        # checkpoint every 5 processed items
        if processed % 5 == 0:
            write_output(args.output, output_container, base, output_hazards_key)
            print(f"[ok] checkpoint saved ({enriched} reasoned so far)")

        if last_err is not None and not already_reasoned(h):
            failed += 1

    write_output(args.output, output_container, base, output_hazards_key)
    print("=== DONE ===")
    print(f"processed={processed} reasoned={enriched} failed={failed}")
    print(f"[ok] wrote: {args.output}")


if __name__ == "__main__":
    main()