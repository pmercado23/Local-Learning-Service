FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY train_ollama_model.py .
COPY training_docs ./training_docs

ENTRYPOINT ["python3", "train_ollama_model.py"]
