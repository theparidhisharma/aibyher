# Project scaffold: aibyher

# inference.py
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

MODEL_DIR = "outputs/lora_qwen_finetuned"
INPUT_FILE = Path("data/filtered/filtered_dataset.jsonl")
OUTPUT_FILE = Path("data/filtered/generated_responses.jsonl")

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()

with open(INPUT_FILE) as f:
    data = [json.loads(l) for l in f]

with torch.no_grad():
    with open(OUTPUT_FILE, "w") as out:
        for sample in tqdm(data):
            prompt = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            output = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.95, temperature=0.7)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            gen = response.split("### Response:")[-1].strip()

            sample["response"] = gen
            out.write(json.dumps(sample) + "\n")

print(f"Saved generated responses to {OUTPUT_FILE}")
