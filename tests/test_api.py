"""
Тесты для FastAPI-сервиса прогнозирования стоимости недвижимости.

Запуск:
    pytest tests/test_api.py -v
"""

import pytest
from httpx import AsyncClient, ASGITransport

# Импортируем приложение
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app


# Тестовые данные (первая запись из датасета)
SAMPLE_PAYLOAD = {
    "bedrooms": 3,
    "bathrooms": 1.0,
    "sqft_living": 1180,
    "sqft_lot": 5650,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1180,
    "sqft_basement": 0,
    "yr_built": 1955,
    "yr_renovated": 0,
    "zipcode": 98178,
    "lat": 47.5112,
    "long": -122.257,
    "sqft_living15": 1340,
    "sqft_lot15": 5650,
}


@pytest.fixture
def client():
    """Создаёт тестовый клиент FastAPI."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """GET /health должен возвращать статус сервиса."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "features" in data


@pytest.mark.asyncio
async def test_predict_endpoint_no_model(client):
    """
    POST /predict без загруженной модели должен вернуть 503.
    """
    response = await client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_predict_endpoint_invalid_data(client):
    """
    POST /predict с некорректными данными должен вернуть 422.
    """
    invalid_payload = {"bedrooms": -1}  # отрицательное значение
    response = await client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_missing_fields(client):
    """
    POST /predict с пропущенными полями должен вернуть 422.
    """
    response = await client.post("/predict", json={"bedrooms": 3})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict_endpoint_no_model(client):
    """
    POST /predict/batch без загруженной модели должен вернуть 503.
    """
    response = await client.post(
        "/predict/batch",
        json={"records": [SAMPLE_PAYLOAD]},
    )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_batch_predict_endpoint_empty(client):
    """
    POST /predict/batch с пустым списком должен вернуть 422.
    """
    response = await client.post(
        "/predict/batch",
        json={"records": []},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_response_schema(client):
    """Проверка схемы ответа /health."""
    response = await client.get("/health")
    data = response.json()
    assert isinstance(data.get("status"), str)
    assert isinstance(data.get("model_loaded"), bool)
    assert isinstance(data.get("features"), list)


@pytest.mark.asyncio
async def test_predict_response_schema(client):
    """Проверка схемы ответа /predict (даже при ошибке 503)."""
    response = await client.post("/predict", json=SAMPLE_PAYLOAD)
    # Сервис без модели возвращает 503, но это корректный JSON
    assert response.headers["content-type"] == "application/json"