import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from utils import get_sentiment_score, is_refusal_response

INPUT_FILE = Path("data/counterfactuals/merged.jsonl")
OUTPUT_FILE = Path("data/filtered/filtered_dataset.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Define simple heuristic slur/profanity patterns
BLOCKLIST = [
    r"\bfuck\b", r"\bnigger\b", r"\bfag\b", r"\bkill yourself\b"
]

def passes_heuristics(text):
    return not any(re.search(p, text.lower()) for p in BLOCKLIST)

def pairwise_invariance_check(samples):
    if len(samples) < 2:
        return [True for _ in samples]
    embeddings = [model.encode(s["response"], convert_to_tensor=True) for s in samples]
    sentiments = [get_sentiment_score(s["response"]) for s in samples]
    refusals = [is_refusal_response(s["response"]) for s in samples]
    base = embeddings[0]
    flags = []
    for i, emb in enumerate(embeddings):
        sim = util.cos_sim(base, emb).item()
        sentiment_delta = abs(sentiments[i] - sentiments[0])
        refusal_delta = int(refusals[i] != refusals[0])
        flags.append(sim >= 0.85 and sentiment_delta <= 0.2 and refusal_delta == 0)
    return flags

def run_filter_pipeline():
    groups = {}
    with open(INPUT_FILE) as f:
        for line in f:
            ex = json.loads(line)
            if ex["counterfactual_group"] not in groups:
                groups[ex["counterfactual_group"]] = []
            groups[ex["counterfactual_group"]].append(ex)

    kept = []
    for group_id, samples in groups.items():
        filtered = []
        for s in samples:
            if not passes_heuristics(s["prompt"] + " " + s["response"]):
                continue
            filtered.append(s)

        flags = pairwise_invariance_check(filtered)
        for i, keep in enumerate(flags):
            if keep:
                kept.append(filtered[i])

    with open(OUTPUT_FILE, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")
    print(f"Filtered down to {len(kept)} samples from {sum(len(v) for v in groups.values())}.")

if __name__ == "__main__":
    run_filter_pipeline()
