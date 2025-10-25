FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set workdir
WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils python3-pip \
        build-essential git ca-certificates cron curl

# Copy dependencies and install
COPY requirements.txt /app/
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . /app
RUN chmod +x /app/run_daily.sh

# Setup cron (runs daily at 2am)
RUN echo "0 * * * * root /app/run_daily.sh >> /var/log/ollama_cron.log 2>&1" > /etc/cron.d/ollama_daily && \
    chmod 0644 /etc/cron.d/ollama_daily && \
    crontab /etc/cron.d/ollama_daily

RUN mkdir -p /var/log && touch /var/log/ollama_cron.log


# Install ollama on local
RUN curl -fsSL https://ollama.com/install.sh | sh

# Default environment variables (override with docker-compose)
ENV BASE_HF_MODEL=Qwen/Qwen2.5-0.5B
ENV OLLAMA_MODEL_NAME=qwen-test-update-01

# Start cron and tail logs
CMD service cron start && tail -F /var/log/ollama_cron.log
