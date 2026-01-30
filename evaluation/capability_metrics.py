# Project scaffold: aibyher

# evaluation/capability_metrics.py
import json
from pathlib import Path
from transformers import pipeline

INPUT_FILE = Path("data/filtered/filtered_dataset.jsonl")
OUTPUT_FILE = Path("data/logs/capability_metrics.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

qa_evaluator = pipeline("question-answering", model="deepset/roberta-base-squad2")

CAPABILITY_TASKS = [
    {
        "context": "The sun is the star at the center of the Solar System.",
        "question": "What is at the center of the Solar System?",
        "answer": "the sun"
    },
    {
        "context": "Water freezes at 0 degrees Celsius.",
        "question": "At what temperature does water freeze?",
        "answer": "0 degrees Celsius"
    },
    {
        "context": "The capital of France is Paris.",
        "question": "What is the capital of France?",
        "answer": "Paris"
    }
]

def run_capability_tests():
    correct = 0
    for task in CAPABILITY_TASKS:
        prediction = qa_evaluator(question=task["question"], context=task["context"])
        pred = prediction["answer"].strip().lower()
        gold = task["answer"].strip().lower()
        if gold in pred or pred in gold:
            correct += 1

    accuracy = correct / len(CAPABILITY_TASKS)
    results = {
        "task_count": len(CAPABILITY_TASKS),
        "correct": correct,
        "accuracy": round(accuracy, 4)
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved capability metrics to {OUTPUT_FILE}.")

if __name__ == "__main__":
    run_capability_tests()