# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim keeps the image small while matching the development Python version.
FROM python:3.11-slim

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY models/ models/   # pre-trained model artifacts (versioned)
COPY app/ app/         # FastAPI application package
COPY train/ train/     # training / registry package (needed by retrain webhook)

EXPOSE 8000

# ── Start API server ──────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
