import json
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("data/logs/bias_metrics.json")

def plot_bias_metrics():
    with open(INPUT_FILE) as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    # Plot intersectional group metrics
    x = []
    sentiment = []
    refusal = []
    for group, vals in metrics["intersectional_groups"].items():
        x.append(group)
        sentiment.append(vals["avg_sentiment"])
        refusal.append(vals["refusal_rate"])

    axes[0].barh(x, sentiment, color='skyblue')
    axes[0].set_title("Average Sentiment per Group")
    axes[0].set_xlabel("Sentiment Score")

    axes[1].barh(x, refusal, color='salmon')
    axes[1].set_title("Refusal Rate per Group")
    axes[1].set_xlabel("Refusal Rate")

    plt.tight_layout()
    plt.savefig("data/logs/group_bias_metrics.png")
    print("Saved group bias metric plots to data/logs/group_bias_metrics.png")

if __name__ == "__main__":
    plot_bias_metrics()
