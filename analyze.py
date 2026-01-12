import json
from collections import Counter
from statistics import mean

PATH = "outputs/hazards.json"

def main():
    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    hazards = data["hazards"]

    # Distributions
    phase = Counter(h.get("task_phase","") for h in hazards)
    sev = Counter(h.get("severity","") for h in hazards)
    cat = Counter(h.get("category","") for h in hazards)
    chain_lens = [len(h.get("propagation_chain", [])) for h in hazards]

    # Simple "vagueness" heuristic
    vague_terms = ["unexpected", "various", "issue", "problem", "may fail", "failure occurs", "not detected", "incorrectly"]
    vagueness_hits = 0
    for h in hazards:
        txt = (h.get("trigger_condition","") + " " + h.get("primary_failure","") + " " + h.get("final_impact","")).lower()
        if any(t in txt for t in vague_terms):
            vagueness_hits += 1

    print("Total hazards:", len(hazards))
    print("\nCategory distribution:", cat)
    print("\nSeverity distribution:", sev)
    print("\nTask phase distribution (top 15):")
    for k,v in phase.most_common(15):
        print(f"  {k}: {v}")

    print("\nPropagation chain length:")
    print("  min:", min(chain_lens), "max:", max(chain_lens), "avg:", round(mean(chain_lens), 2))

    print("\nVagueness heuristic:")
    print("  vagueness_hits:", vagueness_hits, f"({round(vagueness_hits/len(hazards)*100,1)}%)")

if __name__ == "__main__":
    main()