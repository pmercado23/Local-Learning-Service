FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# set a predictable workdir
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         git         ca-certificates         cron         curl         && rm -rf /var/lib/apt/lists/*

# Copy project and install python deps
COPY requirements.txt /app/
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chmod +x /app/run_daily.sh

# Setup cron: run at 02:00 every day
RUN echo "0 2 * * * root /app/run_daily.sh >> /var/log/ollama_cron.log 2>&1" > /etc/cron.d/ollama_daily         && chmod 0644 /etc/cron.d/ollama_daily         && crontab /etc/cron.d/ollama_daily

RUN mkdir -p /var/log

# Default env vars (can be overridden)
ENV BASE_HF_MODEL=meta-llama/Llama-2-7b-chat-hf
ENV OLLAMA_MODEL_NAME=my-custom-model

CMD service cron start && tail -F /var/log/ollama_cron.log
