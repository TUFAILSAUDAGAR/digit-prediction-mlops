import numpy as np
import pytest

from app.inference import DigitPredictor


@pytest.fixture(scope="module")
def predictor():
    p = DigitPredictor()
    p.load()
    return p


class TestDigitPredictor:
    def test_predict_returns_digit(self, predictor):
        img = np.zeros((28, 28), dtype=np.float32)
        meta = {"pen_pressure": 1.0, "writer_age": 25, "handedness": "right"}
        result = predictor.predict(img, meta)
        assert 0 <= result["predicted_digit"] <= 9
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_not_loaded_raises(self):
        p = DigitPredictor()
        with pytest.raises(FileNotFoundError):
            p.predict(
                np.zeros((28, 28)),
                {"pen_pressure": 1.0, "writer_age": 25, "handedness": "right"},
                version="nonexistent",
            )
