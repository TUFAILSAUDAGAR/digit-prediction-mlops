import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_digit_image() -> bytes:
    # Use a non-blank image so it passes the blank-image validation check
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[8:20, 8:20] = 200  # draw a bright square to ensure std > 1e-3
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestMetrics:
    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "request_count" in data
        assert "error_count" in data


class TestPredict:
    def test_valid_prediction(self, client):
        image_bytes = _make_digit_image()
        resp = client.post(
            "/predict",
            files={"image": ("digit.png", image_bytes, "image/png")},
            data={"pen_pressure": 1.0, "writer_age": 25, "handedness": "right"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_digit" in data
        assert 0 <= data["predicted_digit"] <= 9
        assert "confidence" in data

    def test_invalid_handedness(self, client):
        image_bytes = _make_digit_image()
        resp = client.post(
            "/predict",
            files={"image": ("digit.png", image_bytes, "image/png")},
            data={"pen_pressure": 1.0, "writer_age": 25, "handedness": "both"},
        )
        assert resp.status_code == 422

    def test_missing_image(self, client):
        resp = client.post(
            "/predict",
            data={"pen_pressure": 1.0, "writer_age": 25, "handedness": "right"},
        )
        assert resp.status_code == 422

    def test_invalid_pressure(self, client):
        image_bytes = _make_digit_image()
        resp = client.post(
            "/predict",
            files={"image": ("digit.png", image_bytes, "image/png")},
            data={"pen_pressure": -1.0, "writer_age": 25, "handedness": "right"},
        )
        assert resp.status_code == 422

    def test_invalid_image(self, client):
        resp = client.post(
            "/predict",
            files={"image": ("digit.png", b"not an image", "image/png")},
            data={"pen_pressure": 1.0, "writer_age": 25, "handedness": "right"},
        )
        assert resp.status_code == 400
