# Digit Prediction API

A REST API that predicts handwritten digits from images using a CNN model trained on MNIST. Built with FastAPI and PyTorch.

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI application with endpoints
│   ├── models.py         # PyTorch model definitions (CNNEncoder, FinalClassifier)
│   ├── schemas.py        # Pydantic request/response schemas
│   └── inference.py      # Model loading and prediction logic
├── models/               # Trained model artifacts (versioned)
│   └── v1/
│       ├── image_model.pth
│       ├── final_classifier.pth
│       └── metadata_encoder.joblib
├── infra/                # Terraform IaC for AWS deployment
│   └── main.tf
├── tests/
│   ├── test_api.py       # API endpoint tests
│   └── test_inference.py # Inference unit tests
├── digit_prediction.ipynb  # Original training notebook
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── .github/workflows/ci.yml
```

## Setup & Run Locally

### Prerequisites
- Python 3.11+

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict digit from image + metadata (optional `model_version`) |
| `/health` | GET | Health check |
| `/metrics` | GET | Request count, error count, avg latency |
| `/models` | GET | List available model versions |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@example_digit.png" \
  -F "pen_pressure=1.0" \
  -F "writer_age=25" \
  -F "handedness=right"
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Linting

```bash
ruff check app/ tests/
ruff format --check app/ tests/
```

## Docker

### Build

```bash
docker build -t digit-prediction-api .
```

### Run

```bash
docker run -p 8000:8000 digit-prediction-api
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push/PR to `main`:
1. **Lint** — ruff check and format
2. **Test** — pytest
3. **Build** — Docker image build

## Model Versioning

Models are stored in versioned directories under `models/`:

```
models/
├── v1/
│   ├── image_model.pth
│   ├── final_classifier.pth
│   └── metadata_encoder.joblib
└── v2/   (add new versions here)
    └── ...
```

- Pass `model_version=v1` in the predict request to select a version (defaults to `v1`)
- `GET /models` lists all available versions
- New versions are lazy-loaded on first request

## Cloud Deployment (AWS)

Terraform IaC is provided in `infra/` to deploy to AWS ECS Fargate:

```bash
cd infra
terraform init
terraform apply
```

This provisions: ECR repo, ECS Fargate cluster, CloudWatch logging, security group, and a public-facing service.

To deploy:
```bash
# Build and push image
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t digit-prediction-api .
docker tag digit-prediction-api:latest <ecr-repo-url>:latest
docker push <ecr-repo-url>:latest
```

## Model Training

To retrain the model, run the `digit_prediction.ipynb` notebook. It will save model artifacts to the `models/` directory.
