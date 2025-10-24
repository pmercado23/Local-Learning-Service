# train.py

import os
import argparse
from pathlib import Path
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def huggingface_login(token: str = None):
    """
    Logs into Hugging Face to enable model access (e.g., private models or gated LLaMA).
    """
    try:
        from huggingface_hub import login
    except ImportError:
        print("[!] 'huggingface_hub' not installed. Installing now...")
        subprocess.run(["pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import login

    if not token:
        token = os.getenv("HF_TOKEN")
        if not token:
            print("[!] No Hugging Face token provided.")
            print("    Set it via environment variable or pass --hf-token.")
            return False

    try:
        login(token=token)
        print("[+] Logged in to Hugging Face successfully.")
        return True
    except Exception as e:
        print(f"[!] Hugging Face login failed: {e}")
        return False


def load_texts_from_folder(folder):
    texts = []
    for p in Path(folder).glob("**/*"):
        if p.suffix.lower() in {".txt", ".md"}:
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return [{"text": t} for t in texts]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base HF model id (e.g., 'meta-llama/Llama-2-7b-chat-hf')")
    parser.add_argument("--data_dir", required=True, help="Directory with training docs (.txt/.md) or a jsonl file")
    parser.add_argument("--output_dir", default="./output", help="Where to write adapter and artifacts")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--hf_token", type=str, help="Optional Hugging Face token for model access")
    args = parser.parse_args()

     # Login to Hugging Face if token provided
    if args.hf_token or os.getenv("HF_TOKEN"):
        hf_token = args.hf_token if args.hf_token else os.getenv("HF_TOKEN")
        huggingface_login(hf_token)
    else:
        print("[*] Skipping Hugging Face login (no token provided).")

    # Check if Ollama is available
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        print("[!] Ollama not found. Please install it from https://ollama.ai/download or use Docker.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    adapter_dir = Path(args.output_dir) / "adapter"
    adapter_dir.mkdir(exist_ok=True, parents=True)

    # load texts
    if Path(args.data_dir).is_dir():
        ds = Dataset.from_list(load_texts_from_folder(args.data_dir))
    else:
        # assume a jsonl with {"text": "..."}
        ds = load_dataset("json", data_files=args.data_dir, split="train")

    print(f"Loaded dataset with {len(ds)} documents.")

    # tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    tokenized = ds.map(tokenize_function, batched=False, remove_columns=ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")

    # prepare for k-bit training if needed (kept as safe default)
    model = prepare_model_for_kbit_training(model)

    # configure LoRA with PEFT
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    # training args (Trainer works with accelerate when launched via `accelerate launch`)
    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "hf_trainer"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    # save adapter weights in safetensors-compatible format
    print("Saving adapter to:", adapter_dir)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)

    print("Done. Adapter saved to", adapter_dir)

if __name__ == "__main__":
    main()