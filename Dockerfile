FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY telegram_bot.py .
COPY api.py .
COPY repl.py .

COPY examples/ ./examples/

VOLUME ["/app/knowledge_base", "/app/index"]

ENV PYTHONUNBUFFERED=1
ENV INDEX_DIR=/app/index
ENV KB_DIR=/app/knowledge_base

CMD ["python", "telegram_bot.py"]
