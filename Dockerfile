# Base: python:3.11-slim keeps image small
FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY models/ models/   # pre-trained model artifacts
COPY app/ app/         # FastAPI app
COPY train/ train/     # needed by retrain webhook

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
