#!/usr/bin/env python3
import os
import json
import time
import argparse
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# OpenAI Python SDK (v1+)
from openai import OpenAI

load_dotenv()

IN_PATH_DEFAULT = "outputs/hazards_normalized.json"
OUT_PATH_DEFAULT = "outputs/hazards_enriched.json"

DEFAULT_MODEL = os.getenv("MODEL", "gpt-4.1-mini")

# -------------- Utilities --------------

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n", "", s)
        s = re.sub(r"\n```$", "", s).strip()
    return s

def try_parse_json(s: str) -> Any:
    return json.loads(strip_code_fences(s))

def get_output_text(resp: Any) -> str:
    # Newer SDKs
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text

    # Some SDKs expose `output` as a list of items with `content`
    if hasattr(resp, "output") and isinstance(resp.output, list):
        chunks = []
        for item in resp.output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for c in content:
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str):
                        chunks.append(txt)
            elif isinstance(content, str):
                chunks.append(content)
        txt_joined = "\n".join(chunks).strip()
        if txt_joined:
            return txt_joined

    # Fallback: best-effort stringify
    return str(resp)

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    write_json(tmp, obj)
    os.replace(tmp, path)

def chunk_list(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def is_enriched(h: Dict[str, Any]) -> bool:
    return (
        isinstance(h.get("observables"), list)
        and isinstance(h.get("mitigations"), list)
        and isinstance(h.get("verification_idea"), str)
        and isinstance(h.get("safety_requirement"), str)
    )

def clamp_list_str(xs: Any, min_len: int, max_len: int) -> List[str]:
    if not isinstance(xs, list):
        return []
    out = []
    for x in xs:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
    # de-dup while preserving order
    seen = set()
    deduped = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    if len(deduped) < min_len:
        return deduped
    return deduped[:max_len]

def sanitize_text(s: Any, max_chars: int = 400) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "..."
    return s

# -------------- Prompt + Schema --------------

def build_context_blob(data: Dict[str, Any]) -> str:
    # Keep it compact but informative.
    system_input = sanitize_text(data.get("system_input", ""), 900)
    components = data.get("components", {}) or {}
    task = data.get("task", {}) or {}

    # Flatten components for easy reading
    def join_list(key: str) -> str:
        v = components.get(key, [])
        if isinstance(v, list):
            return ", ".join([str(x) for x in v])
        return ""

    sensors = join_list("sensors")
    actuators = join_list("actuators")
    compute = join_list("compute_software")
    control = join_list("control_modules")
    hmi = join_list("human_interfaces")

    phases = task.get("task_phases", [])
    if isinstance(phases, list):
        phases_s = "; ".join(phases)
    else:
        phases_s = ""

    failures = task.get("notable_failure_outcomes", [])
    if isinstance(failures, list):
        failures_s = "; ".join(failures)
    else:
        failures_s = ""

    ctx = f"""SYSTEM SUMMARY (do not rewrite, only use as reference):
{system_input}

COMPONENTS (canonical names):
- sensors: {sensors}
- actuators: {actuators}
- compute_software: {compute}
- control_modules: {control}
- human_interfaces: {hmi}

TASK PHASES:
{phases_s}

NOTABLE FAILURE OUTCOMES:
{failures_s}
"""
    return ctx

def build_user_prompt(context_blob: str, hazard: Dict[str, Any]) -> str:
    # Only pass what matters for enrichment.
    hazard_short = {
        "hazard_id": hazard.get("hazard_id"),
        "category": hazard.get("category"),
        "subtype": hazard.get("subtype"),
        "trigger_condition": hazard.get("trigger_condition"),
        "primary_failure": hazard.get("primary_failure"),
        "propagation_chain": hazard.get("propagation_chain"),
        "final_impact": hazard.get("final_impact"),
        "severity": hazard.get("severity"),
        "likelihood": hazard.get("likelihood"),
        "detectability": hazard.get("detectability"),
        "affected_components": hazard.get("affected_components"),
        "task_phase": hazard.get("task_phase"),
        "notes": hazard.get("notes", ""),
    }

    return f"""{context_blob}

You will enrich a robotics hazard record into actionable engineering outputs.

RULES:
- Output MUST be in English.
- Output MUST be ONLY valid JSON (no markdown, no prose).
- Output must match this exact JSON shape: {{"hazard_id": string, "observables": [3-6 strings], "mitigations": [3-6 strings], "verification_idea": string, "safety_requirement": string}}
- Do NOT change any existing fields; only generate enrichment fields.
- Keep observables measurable/signals/logs/metrics (not vague).
- Mitigations should be implementable (design controls, software guards, procedures).
- verification_idea must describe a concrete test/injection scenario and what to check.
- safety_requirement must be a single sentence using "IF ... THEN ... MUST ..." style, with a measurable threshold placeholder if needed (e.g., X ms, Y kPa).
- Use only components that plausibly exist in the system; prefer affected_components when possible.
- Avoid generic fluff.

INPUT HAZARD (JSON):
{json.dumps(hazard_short, ensure_ascii=False, indent=2)}
"""

def build_json_repair_prompt(bad_output: str, error: str) -> str:
    return f"""You fix invalid JSON. Output JSON only, nothing else.

STRICT RULES:
- Return ONLY JSON. No markdown. No explanations.
- Do not wrap the JSON in code fences.

The following text was intended to be JSON but failed to parse.
Fix it and return ONLY valid JSON.

Parse error:
{error}

Text:
{bad_output}
""".strip()

# -------------- OpenAI call --------------

def call_enrich(client: OpenAI, model: str, context_blob: str, hazard: Dict[str, Any],
                max_retries: int = 4, sleep_base: float = 1.2) -> Dict[str, Any]:
    user_prompt = build_user_prompt(context_blob, hazard)

    last_text: Optional[str] = None

    for attempt in range(max_retries):
        try:
            if last_text is None:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": "You are a senior robotics safety engineer. Be specific and test-oriented."},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.25,
                )
                last_text = get_output_text(resp)
            else:
                # Repair attempt
                repair_prompt = build_json_repair_prompt(last_text, "JSON parse error")
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": "You fix invalid JSON and output JSON only."},
                        {"role": "user", "content": repair_prompt},
                    ],
                    temperature=0.0,
                )
                last_text = get_output_text(resp)

            obj = try_parse_json(last_text)

            # Basic post-validate/sanitize
            hid = str(obj.get("hazard_id", "")).strip()
            if hid != str(hazard.get("hazard_id", "")).strip():
                raise ValueError(f"hazard_id mismatch: got={hid} expected={hazard.get('hazard_id')}")

            obj["observables"] = clamp_list_str(obj.get("observables"), 3, 6)
            obj["mitigations"] = clamp_list_str(obj.get("mitigations"), 3, 6)
            obj["verification_idea"] = sanitize_text(obj.get("verification_idea"), 400)
            obj["safety_requirement"] = sanitize_text(obj.get("safety_requirement"), 240)

            if len(obj["observables"]) < 3 or len(obj["mitigations"]) < 3:
                raise ValueError("observables/mitigations too short after sanitization")
            if len(obj["verification_idea"]) < 20 or len(obj["safety_requirement"]) < 20:
                raise ValueError("verification_idea/safety_requirement too short")

            return obj

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                # Show a snippet to debug
                snippet = (last_text or "")[:600]
                raise json.JSONDecodeError(f"{e.msg}. Model output head: {snippet}", e.doc, e.pos)
            # next attempt will run the repair branch
            time.sleep(sleep_base * (2 ** attempt))
            continue
        except Exception as e:
            if last_text is None:
                # If we failed before getting any model output, force a fresh generation next loop
                last_text = None
            if attempt == max_retries - 1:
                raise
            time.sleep(sleep_base * (2 ** attempt))
            # continue to allow another attempt (repair or regen)
            continue

    raise RuntimeError("Unreachable")

# -------------- Main pipeline --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=IN_PATH_DEFAULT)
    ap.add_argument("--out", dest="out_path", default=OUT_PATH_DEFAULT)
    ap.add_argument("--model", dest="model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", dest="limit", type=int, default=0, help="Process only N hazards (0 = all)")
    ap.add_argument("--start", dest="start", type=int, default=0, help="Start index in hazards list")
    ap.add_argument("--batch", dest="batch", type=int, default=1, help="Batch size (keep 1 for stable quality)")
    ap.add_argument("--resume", dest="resume", action="store_true", help="Resume from existing out file")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env")

    client = OpenAI(api_key=api_key)

    data = read_json(args.in_path)
    hazards: List[Dict[str, Any]] = data.get("hazards", []) or []
    if not hazards:
        raise RuntimeError("No hazards found in input JSON")

    # Resume support
    out_data: Dict[str, Any]
    if args.resume and os.path.exists(args.out_path):
        out_data = read_json(args.out_path)
        out_hazards = out_data.get("hazards", []) or []
        # Build index by hazard_id
        out_by_id = {h.get("hazard_id"): h for h in out_hazards if isinstance(h, dict)}
    else:
        out_data = dict(data)
        out_by_id = {}

    context_blob = build_context_blob(data)

    # Determine slice
    start = max(0, args.start)
    end = len(hazards)
    if args.limit and args.limit > 0:
        end = min(end, start + args.limit)
    work = hazards[start:end]

    total = len(work)
    print(f"[info] model={args.model}")
    print(f"[info] input={args.in_path}")
    print(f"[info] output={args.out_path}")
    print(f"[info] hazards total={len(hazards)} processing={total} (start={start}, end={end})")
    print(f"[info] resume={'yes' if args.resume else 'no'} batch={args.batch} dry_run={'yes' if args.dry_run else 'no'}")

    processed = 0
    enriched_count = 0
    skipped = 0

    # Process (batch currently kept simple; batch>1 means sequential anyway)
    for i, h in enumerate(work, start=start):
        hid = h.get("hazard_id")
        if not hid:
            print(f"[warn] missing hazard_id at index={i}, skipping")
            skipped += 1
            continue

        # If already enriched in output, skip
        if hid in out_by_id and is_enriched(out_by_id[hid]):
            skipped += 1
            processed += 1
            continue

        if args.dry_run:
            print(f"[dry] would enrich {hid}")
            processed += 1
            continue

        try:
            enr = call_enrich(client, args.model, context_blob, h)
            # Merge into hazard (do not delete existing)
            h_new = dict(h)
            h_new["observables"] = enr["observables"]
            h_new["mitigations"] = enr["mitigations"]
            h_new["verification_idea"] = enr["verification_idea"]
            h_new["safety_requirement"] = enr["safety_requirement"]

            out_by_id[hid] = h_new
            enriched_count += 1
            processed += 1

            # Periodic save (every 5 enrichments)
            if enriched_count % 5 == 0:
                merged_list = []
                # preserve original ordering from input hazards
                for hh in hazards:
                    hh_id = hh.get("hazard_id")
                    if hh_id in out_by_id:
                        merged_list.append(out_by_id[hh_id])
                    else:
                        merged_list.append(hh)
                out_data["hazards"] = merged_list
                out_data["enrichment_meta"] = {
                    "model": args.model,
                    "timestamp_unix": int(time.time()),
                    "fields_added": ["observables", "mitigations", "verification_idea", "safety_requirement"],
                }
                atomic_write_json(args.out_path, out_data)
                print(f"[ok] checkpoint saved ({enriched_count} enriched)")

            print(f"[ok] enriched {hid}")

        except Exception as e:
            print(f"[err] failed to enrich {hid}: {type(e).__name__}: {e}")
            print("[hint] You can retry with --resume to continue from last checkpoint.")
            # Keep going; you can re-run with --resume
            processed += 1

    # Final save
    if not args.dry_run:
        merged_list = []
        for hh in hazards:
            hh_id = hh.get("hazard_id")
            if hh_id in out_by_id:
                merged_list.append(out_by_id[hh_id])
            else:
                merged_list.append(hh)
        out_data["hazards"] = merged_list
        out_data["enrichment_meta"] = {
            "model": args.model,
            "timestamp_unix": int(time.time()),
            "fields_added": ["observables", "mitigations", "verification_idea", "safety_requirement"],
        }
        atomic_write_json(args.out_path, out_data)

    print(f"[done] processed={processed} enriched={enriched_count} skipped={skipped}")
    if args.dry_run:
        print("[done] dry-run only; no files written.")

if __name__ == "__main__":
    main()