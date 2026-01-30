import os
import json
import random
from pathlib import Path
from utils import fill_template
from templates import (
hiring_template, health_template, moderation_template,
education_template, finance_template
)
from configs.attributes import attributes

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

DOMAIN_TEMPLATES = {
    "hiring": hiring_template,
    "health": health_template,
    "moderation": moderation_template,
    "education": education_template,
    "finance": finance_template
}

SAMPLES_PER_DOMAIN = 1000

def generate_dataset():
    all_samples = []
    group_counter = 0
    for domain, template in DOMAIN_TEMPLATES.items():
        for _ in range(SAMPLES_PER_DOMAIN):
            sample = fill_template(template, attributes)
            sample_record = {
                "id": sample["id"],
                "task_domain": domain,
                "task_type": "generation",
                "prompt": sample["prompt"],
                "response": "<TO_BE_GENERATED>",
                "sensitive_attributes": sample["attributes"],
                "counterfactual_group": f"cf_{group_counter:05d}",
                "counterfactual_axis": None,
                "source": "synthetic",
                "generation_seed": random.randint(1000000, 9999999),
                "filtering_metadata": None
            }
            group_counter += 1
            all_samples.append(sample_record)

    with open(output_dir / "initial_dataset.jsonl", "w") as f:
        for record in all_samples:
            f.write(json.dumps(record) + "\n")

    print(f"Generated {len(all_samples)} samples across {len(DOMAIN_TEMPLATES)} domains.")

if __name__ == "__main__":
    generate_dataset()
