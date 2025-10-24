#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$ROOT_DIR/output"
ADAPTER_DIR="$OUTPUT_DIR/adapter"
MODELFILENAME="Modelfile"
MODELFILENAME_PATH="$ROOT_DIR/$MODELFILENAME"
MODELNAME=${OLLAMA_MODEL_NAME:-"qwen-test-update-01"}
BASE_MODEL=${BASE_HF_MODEL:-"Qwen/Qwen2.5-0.5B"}  # override via env
HF_TOKEN=${HF_TOKEN}

echo "Starting daily run: $(date)"

# 1) Run training (use accelerate if available)
if command -v accelerate >/dev/null 2>&1; then
  echo "Running training with accelerate launch (if configured)."
  accelerate launch /app/train.py --model "$BASE_MODEL" --data_dir /app/data --output_dir "$OUTPUT_DIR" --num_train_epochs 3 || {
    echo "Training failed (nonfatal). Continuing to attempt to import adapter if present."
  }
else
  python3 /app/train.py --model "$BASE_MODEL" --data_dir /app/data --output_dir "$OUTPUT_DIR" --num_train_epochs 3|| {
    echo "Training failed (nonfatal). Continuing to attempt to import adapter if present."
  }
fi

if [ ! -d "$ADAPTER_DIR" ]; then
  echo "Adapter directory not found at $ADAPTER_DIR. Exiting with non-zero."
  exit 1
fi

# 2) Render Modelfile
cat /app/Modelfile.template       | sed "s|{{BASE_MODEL}}|$BASE_MODEL|g"       | sed "s|{{ADAPTER_PATH}}|$ADAPTER_DIR|g"       > "$MODELFILENAME_PATH"

echo "Wrote Modelfile:"
cat "$MODELFILENAME_PATH"

# 3) Create/update Ollama model
echo "stating ollama" 
ollama serve & 

if command -v ollama >/dev/null 2>&1; then
  echo "Ollama CLI found. Creating/updating model: $MODELNAME"
  if ollama ls | grep -q "$MODELNAME"; then
    echo "Model $MODELNAME exists; removing then re-creating"
    ollama rm "$MODELNAME" || true
  fi
  ollama create "$MODELNAME" -f "$MODELFILENAME_PATH"
  echo "Created Ollama model: $MODELNAME"
else
  echo "ollama CLI not found in PATH. Please install Ollama on the host or add it to the container."
  exit 2
fi

echo "Daily run complete: $(date)"
