# Project scaffold: aibyher

# training/lora_train.py
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

CONFIG_PATH = Path("training/config_lora.json")
DATA_PATH = Path("data/filtered/filtered_dataset.jsonl")


def load_jsonl_dataset(path):
    with open(path) as f:
        data = [json.loads(l) for l in f]
    return Dataset.from_list([{
        "prompt": d["prompt"],
        "response": d["response"] if "response" in d else "<TO_BE_GENERATED>"
    } for d in data])

def tokenize(example, tokenizer, max_len):
    prompt = example["prompt"]
    response = example["response"]
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

def main():
    config = json.load(open(CONFIG_PATH))

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    dataset = load_jsonl_dataset(DATA_PATH)
    tokenized = dataset.map(lambda x: tokenize(x, tokenizer, config["max_seq_len"]), remove_columns=["prompt", "response"])

    training_args = TrainingArguments(
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        output_dir=config["output_dir"],
        report_to=config["report_to"],
        fp16=config["fp16"]
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        args=training_args,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    main()