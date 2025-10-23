# Ollama HF Train (with Accelerate) - Daily Cron

This project fine-tunes a Hugging Face model using PEFT/LoRA, exports a safetensors adapter, and imports it into Ollama via a Modelfile. It includes an `accelerate` config to run distributed GPU training and a Docker setup that runs the pipeline once a day via cron.

## Features added in this version
- `accelerate_config.yaml` provided for distributed GPU training.
- `run_daily.sh` will use `accelerate launch` if `accelerate` is installed.
- Dockerfile uses an NVIDIA CUDA runtime image; use `docker run --gpus all` or `docker-compose` with `runtime: nvidia`.

## Quick start (local GPU, interactive)
1. Put your training docs in `./data` (plain text files, one file per doc) or a single JSONL with `{"text": "..."}`
2. Configure accelerate (optional but recommended for multi-GPU):
   ```bash
   pip install accelerate
   accelerate config
   # or use the provided accelerate_config.yaml by running:
   accelerate config default --config_file accelerate_config.yaml
   ```
3. Start training with accelerate:
   ```bash
   accelerate launch train.py --model <HF_MODEL> --data_dir ./data --output_dir ./output
   ```
   Example:
   ```bash
   accelerate launch train.py --model meta-llama/Llama-2-7b-chat-hf --data_dir ./data --output_dir ./output --num_train_epochs 1
   ```

## Running in Docker
- Build:
  ```bash
  docker build -t ollama-hf-train .
  ```
- Run (with GPU access):
  ```bash
  docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output ollama-hf-train
  ```
  Ensure Ollama CLI is available either in the container or mount host's Ollama directory.

## Using the Modelfile
The `run_daily.sh` script renders `Modelfile.template` to `Modelfile` pointing to the adapter and runs:
```
ollama create <modelname> -f Modelfile
```
Ensure `ollama` CLI is installed and accessible.

## accelerate_config.yaml
A recommended `accelerate` config is included (`accelerate_config.yaml`). It targets a multi-GPU setup using `deepspeed`/`fp16` style training. Adjust as needed for your cluster.

## Notes & caveats
- Fine-tuning large models requires sufficient GPU RAM; use gradient accumulation, 8/4-bit training (bitsandbytes), or Deepspeed stage 3 for very large models.
- The Docker image uses CUDA base — ensure host drivers match.
- Ollama import requires adapter safetensors or GGUF. This repo exports safetensors adapters by default.
