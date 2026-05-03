FROM python:3.11-slim

WORKDIR /app

# Install build deps for packages that compile C extensions (chromadb, aiosqlite)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first (better layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Ingest docs on first run, then start server
CMD ["sh", "-c", "python -m app.rag.ingest --path docs/ && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
