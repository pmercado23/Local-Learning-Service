# 🧠 Ollama Trainer

A lightweight Dockerized project to create custom Ollama models from your documents.

## 🚀 Features
- Extend any Ollama base model with your docs.
- Auto-generates a `Modelfile` and builds a new custom model.
- Dockerized for easy, reproducible setup.

## 🧱 Project Structure
```
ollama-trainer/
├── Dockerfile
├── requirements.txt
├── train_ollama_model.py
├── README.md
└── training_docs/
```

## ⚙️ Requirements
- Docker installed and running  
- [Ollama](https://ollama.ai/download) installed locally **if running outside Docker**

## 🐳 Run with Docker

### 1️⃣ Build the image
```bash
docker build -t ollama-trainer .
```

### 2️⃣ Run the container
```bash
docker run -it --rm   -v ollama:/root/.ollama   -v $(pwd)/training_docs:/app/training_docs   ollama-trainer   --base llama3   --name mydocs_model   --docs ./training_docs
```

### 3️⃣ Verify the model
```bash
ollama run mydocs_model
```
