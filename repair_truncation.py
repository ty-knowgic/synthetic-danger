#!/usr/bin/env python3
# repair_truncation.py
#
# Repair truncated fields in outputs/hazards_enriched.json:
# - safety_requirement
# - verification_idea
#
# It detects "..." or "…" and regenerates only those fields via OpenAI.

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI  # pip install openai
from dotenv import load_dotenv  # pip install python-dotenv

TRUNC_PAT = re.compile(r"(\.\.\.|…)")


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def is_truncated(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    return bool(TRUNC_PAT.search(s.strip()))


def clamp_whitespace(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def build_prompt(system_input: str, hazard: Dict[str, Any]) -> str:
    # Keep context short: focus on the hazard fields needed for regeneration.
    # The model will produce JSON with two fields.
    return f"""
You are a safety engineer. Rewrite ONLY the two fields below with complete, non-truncated text.

Constraints:
- Output must be STRICT JSON (no markdown, no commentary).
- Keys: safety_requirement, verification_idea
- safety_requirement: one line, measurable, uses IF/THEN/MUST style, includes explicit variables like X ms / Y kPa as placeholders (do NOT invent exact numeric values).
- verification_idea: 3-6 sentences, concrete fault-injection or test procedure, includes what signals/logs to check.
- Do NOT include "..." or "…" anywhere.
- Keep safety_requirement <= 360 characters.
- Keep verification_idea <= 900 characters.

System context:
{system_input}

Hazard (do not change other fields):
hazard_id: {hazard.get("hazard_id")}
category: {hazard.get("category")}
subtype: {hazard.get("subtype")}
task_phase: {hazard.get("task_phase")}
trigger_condition: {hazard.get("trigger_condition")}
primary_failure: {hazard.get("primary_failure")}
propagation_chain: {hazard.get("propagation_chain")}
final_impact: {hazard.get("final_impact")}
affected_components: {hazard.get("affected_components")}
observables: {hazard.get("observables")}
mitigations: {hazard.get("mitigations")}

Current (possibly truncated):
safety_requirement: {hazard.get("safety_requirement")}
verification_idea: {hazard.get("verification_idea")}
""".strip()


def call_repair(
    client: OpenAI,
    model: str,
    system_input: str,
    hazard: Dict[str, Any],
    max_retries: int = 5,
) -> Tuple[str, str]:
    prompt = build_prompt(system_input, hazard)

    sleep_base = 1.0
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            text = resp.choices[0].message.content.strip()
            data = json.loads(text)

            sr = clamp_whitespace(str(data["safety_requirement"]))
            vi = clamp_whitespace(str(data["verification_idea"]))

            # Guardrails
            if is_truncated(sr) or is_truncated(vi):
                raise ValueError("Model returned truncated markers")
            if len(sr) > 360:
                raise ValueError(f"safety_requirement too long ({len(sr)})")
            if len(vi) > 900:
                raise ValueError(f"verification_idea too long ({len(vi)})")

            return sr, vi

        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** attempt))

    raise RuntimeError(f"Failed to repair {hazard.get('hazard_id')}: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/hazards_enriched.json")
    ap.add_argument("--output", default="outputs/hazards_enriched_repaired.json")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--limit", type=int, default=0, help="Repair at most N hazards (0=all)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    doc = load_json(in_path)
    hazards: List[Dict[str, Any]] = doc.get("hazards", [])
    system_input = str(doc.get("system_input", "")).strip()

    # detect targets
    targets = []
    for i, h in enumerate(hazards):
        sr = h.get("safety_requirement")
        vi = h.get("verification_idea")
        if is_truncated(sr) or is_truncated(vi):
            targets.append(i)

    if args.limit and len(targets) > args.limit:
        targets = targets[: args.limit]

    print("=== REPAIR PLAN ===")
    print(f"input:  {in_path}")
    print(f"output: {out_path}")
    print(f"model:  {args.model}")
    print(f"hazards_total: {len(hazards)}")
    print(f"targets: {len(targets)}")
    if targets:
        print("target_ids:", ", ".join(hazards[i].get("hazard_id", "?") for i in targets))

    if not targets:
        print("[ok] no truncated fields detected. Nothing to do.")
        if not args.dry_run:
            # still write a copy, for clarity
            save_json(out_path, doc)
            print(f"[ok] wrote: {out_path}")
        return

    if args.dry_run:
        print("[dry_run] exiting without changes.")
        return

    # Load environment variables from .env (same convention as run.py/enrich.py)
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env or export it in your shell before running."
        )

    client = OpenAI(api_key=api_key)

    repaired = 0
    for idx in targets:
        h = hazards[idx]
        hid = h.get("hazard_id", "?")
        print(f"[info] repairing {hid} ...")
        sr_new, vi_new = call_repair(client, args.model, system_input, h)
        h["safety_requirement"] = sr_new
        h["verification_idea"] = vi_new
        h["repair_meta"] = {
            "repaired_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "fields": ["safety_requirement", "verification_idea"],
        }
        repaired += 1

    doc["hazards"] = hazards
    doc["repair_summary"] = {
        "repaired_count": repaired,
        "targets_count": len(targets),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    save_json(out_path, doc)
    print(f"[ok] wrote: {out_path}")
    print(f"[done] repaired={repaired} / targets={len(targets)}")


if __name__ == "__main__":
    main()