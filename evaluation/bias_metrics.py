# Project scaffold: aibyher

# evaluation/bias_metrics.py
import json
from pathlib import Path
from collections import defaultdict
from utils import get_sentiment_score, is_refusal_response

INPUT_FILE = Path("data/filtered/filtered_dataset.jsonl")
OUTPUT_FILE = Path("data/logs/bias_metrics.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TARGET_ATTRIBUTES = ["gender", "religion", "nationality"]


def compute_bias_metrics():
    sentiment_by_group = defaultdict(list)
    refusal_by_group = defaultdict(list)
    group_counts = defaultdict(int)

    sentiment_by_attr = {attr: defaultdict(list) for attr in TARGET_ATTRIBUTES}
    refusal_by_attr = {attr: defaultdict(list) for attr in TARGET_ATTRIBUTES}

    with open(INPUT_FILE) as f:
        for line in f:
            ex = json.loads(line)
            attrs = ex["sensitive_attributes"]
            group_key = "_".join([attrs.get(attr, "") for attr in TARGET_ATTRIBUTES])
            group_counts[group_key] += 1

            sentiment = get_sentiment_score(ex["response"])
            refusal = is_refusal_response(ex["response"])

            sentiment_by_group[group_key].append(sentiment)
            refusal_by_group[group_key].append(int(refusal))

            for attr in TARGET_ATTRIBUTES:
                val = attrs.get(attr)
                if val:
                    sentiment_by_attr[attr][val].append(sentiment)
                    refusal_by_attr[attr][val].append(int(refusal))

    results = {
        "intersectional_groups": {},
        "attribute_disparities": {}
    }

    for group in sentiment_by_group:
        avg_sentiment = sum(sentiment_by_group[group]) / len(sentiment_by_group[group])
        refusal_rate = sum(refusal_by_group[group]) / len(refusal_by_group[group])
        results["intersectional_groups"][group] = {
            "avg_sentiment": round(avg_sentiment, 4),
            "refusal_rate": round(refusal_rate, 4),
            "count": group_counts[group]
        }

    for attr in TARGET_ATTRIBUTES:
        results["attribute_disparities"][attr] = {}
        for val, vals in sentiment_by_attr[attr].items():
            avg_sent = sum(vals) / len(vals)
            avg_refuse = sum(refusal_by_attr[attr][val]) / len(refusal_by_attr[attr][val])
            results["attribute_disparities"][attr][val] = {
                "avg_sentiment": round(avg_sent, 4),
                "refusal_rate": round(avg_refuse, 4),
                "count": len(vals)
            }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved full bias metric breakdown to {OUTPUT_FILE}.")

if __name__ == "__main__":
    compute_bias_metrics()
