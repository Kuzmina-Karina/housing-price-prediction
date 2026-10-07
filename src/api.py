"""
FastAPI-сервис для инференса модели прогнозирования стоимости недвижимости.

Эндпоинты:
- GET  /health — проверка состояния сервиса
- POST /predict — предсказание для одного объекта
- POST /predict/batch — пакетное предсказание
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predictor
from src.utils import setup_logging, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic-схемы
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Запрос на предсказание для одного объекта."""

    bedrooms: int = Field(..., ge=0, description="Количество спален")
    bathrooms: float = Field(..., ge=0, description="Количество ванных комнат")
    sqft_living: int = Field(..., ge=0, description="Жилая площадь (кв. футы)")
    sqft_lot: int = Field(..., ge=0, description="Площадь участка (кв. футы)")
    floors: float = Field(..., ge=0, description="Количество этажей")
    waterfront: int = Field(..., ge=0, le=1, description="Вид на воду (0/1)")
    view: int = Field(..., ge=0, le=4, description="Индекс вида (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Состояние (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Рейтинг конструкции (1-13)")
    sqft_above: int = Field(..., ge=0, description="Площадь над землёй (кв. футы)")
    sqft_basement: int = Field(..., ge=0, description="Площадь подвала (кв. футы)")
    yr_built: int = Field(..., ge=1800, le=2025, description="Год постройки")
    yr_renovated: int = Field(..., ge=0, description="Год ремонта (0 = нет ремонта)")
    zipcode: int = Field(..., ge=98000, le=98200, description="Почтовый индекс")
    lat: float = Field(..., ge=47.0, le=48.0, description="Широта")
    long: float = Field(..., ge=-123.0, le=-121.0, description="Долгота")
    sqft_living15: int = Field(..., ge=0, description="Средняя жилая площадь 15 соседей (кв. футы)")
    sqft_lot15: int = Field(..., ge=0, description="Средняя площадь участка 15 соседей (кв. футы)")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.25,
                "sqft_living": 2570,
                "sqft_lot": 7242,
                "floors": 2.0,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "grade": 7,
                "sqft_above": 2170,
                "sqft_basement": 400,
                "yr_built": 1951,
                "yr_renovated": 1991,
                "zipcode": 98125,
                "lat": 47.7210,
                "long": -122.319,
                "sqft_living15": 1690,
                "sqft_lot15": 7639,
            }
        }

    def to_features_dict(self) -> dict[str, float]:
        """Преобразует запрос в словарь для модели."""
        return {
            "bedrooms": float(self.bedrooms),
            "bathrooms": self.bathrooms,
            "sqft_living": float(self.sqft_living),
            "sqft_lot": float(self.sqft_lot),
            "floors": self.floors,
            "waterfront": float(self.waterfront),
            "view": float(self.view),
            "condition": float(self.condition),
            "grade": float(self.grade),
            "sqft_above": float(self.sqft_above),
            "sqft_basement": float(self.sqft_basement),
            "yr_built": float(self.yr_built),
            "yr_renovated": float(self.yr_renovated),
            "zipcode": float(self.zipcode),
            "lat": self.lat,
            "long": self.long,
            "sqft_living15": float(self.sqft_living15),
            "sqft_lot15": float(self.sqft_lot15),
        }


class PredictResponse(BaseModel):
    """Ответ с предсказанием."""

    price: float = Field(..., description="Предсказанная стоимость недвижимости (USD)")
    currency: str = Field("USD", description="Валюта")


class BatchPredictRequest(BaseModel):
    """Запрос на пакетное предсказание."""

    records: list[PredictRequest] = Field(
        ..., min_length=1, max_length=1000,
        description="Список объектов для предсказания"
    )


class BatchPredictResponse(BaseModel):
    """Ответ на пакетное предсказание."""

    predictions: list[float] = Field(
        ..., description="Список предсказанных цен"
    )
    currency: str = Field("USD", description="Валюта")


class HealthResponse(BaseModel):
    """Ответ на health-check."""

    status: str = Field("ok", description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    features: list[str] = Field(
        default_factory=list, description="Список признаков модели"
    )


# ---------------------------------------------------------------------------
# Приложение FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель при старте приложения."""
    logger.info("Загрузка модели...")
    try:
        predictor.load()
        logger.info("Модель успешно загружена")
    except FileNotFoundError as e:
        logger.warning("Модель не найдена: %s. Сервис работает без модели.", e)
    yield


config = load_config()
api_config = config.get("api", {})
setup_logging(config.get("logging", {}))

app = FastAPI(
    title=api_config.get("title", "USA Housing Price Prediction API"),
    description="REST API для инференса модели прогнозирования стоимости недвижимости",
    version=api_config.get("version", "1.0.0"),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Проверка состояния сервиса."""
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        features=predictor.features if predictor.is_loaded else [],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Предсказать стоимость недвижимости для одного объекта.

    Тело запроса содержит 18 признаков датасета King County House Sales.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        features = request.to_features_dict()
        price = predictor.predict(features)
        logger.info("Prediction: %s -> %.2f", features, price)
        return PredictResponse(price=round(price, 2))
    except Exception as e:
        logger.error("Ошибка предсказания: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Inference"],
)
def predict_batch(request: BatchPredictRequest):
    """
    Пакетное предсказание для нескольких объектов (до 1000).
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        records = [r.to_features_dict() for r in request.records]
        prices = predictor.predict_batch(records)
        logger.info("Batch prediction: %d records", len(records))
        return BatchPredictResponse(
            predictions=[round(p, 2) for p in prices]
        )
    except Exception as e:
        logger.error("Ошибка пакетного предсказания: %s", e)
        raise HTTPException(status_code=500, detail=str(e))