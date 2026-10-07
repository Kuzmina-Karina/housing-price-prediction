"""Проверка сохранённой модели и модели, создаваемой ноутбуком."""

import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingRegressor

from src import api as api_module
from src.predict import Predictor
from tests.test_api import SAMPLE_PAYLOAD


def test_api_with_saved_model(monkeypatch):
    predictor = Predictor()
    monkeypatch.setattr(api_module, "predictor", predictor)

    with TestClient(api_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        single = client.post("/predict", json=SAMPLE_PAYLOAD)
        batch = client.post("/predict/batch", json={"records": [SAMPLE_PAYLOAD]})
        assert single.status_code == batch.status_code == 200

        model = joblib.load(predictor.model_path)
        expected = float(model.predict(pd.DataFrame([SAMPLE_PAYLOAD]))[0])
        assert np.isfinite(expected) and expected > 0
        assert single.json()["price"] == pytest.approx(expected, abs=0.01)
        assert batch.json()["predictions"][0] == single.json()["price"]


def test_notebook_pickle_model_restores_price(tmp_path):
    X = pd.DataFrame({"sqft_living": [1000, 1500, 2000, 2500]})
    prices = np.array([150000, 225000, 300000, 375000])
    model = HistGradientBoostingRegressor(
        max_iter=2, min_samples_leaf=1, random_state=42
    ).fit(X, np.log1p(prices))

    model_path = tmp_path / "model.pkl"
    features_path = tmp_path / "model_features.pkl"
    model_path.write_bytes(pickle.dumps(model))
    features_path.write_bytes(pickle.dumps(list(X.columns)))

    predictor = Predictor()
    predictor.model_path = model_path
    predictor.features_path = features_path
    predictor.load()

    record = {"sqft_living": 1500}
    expected = float(np.expm1(model.predict(pd.DataFrame([record]))[0]))
    assert predictor.predict(record) == pytest.approx(expected)
    assert predictor.predict_batch([record]) == pytest.approx([expected])
