import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

from src.utils import load_config

logger = logging.getLogger(__name__)


class Predictor:
    """Загрузчик и обёртка для модели регрессии."""

    def __init__(self, config_path: str = "config.yaml"):
            config = load_config(config_path)
            self.model_path = Path("models/model.pkl")
            self.features_path = Path("models/model_features.pkl")
            self._model = None
            self._features: list[str] = []

    def load(self) -> None:
        """Загружает модель и признаки, сохранённые через joblib или pickle."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Файл признаков не найден: {self.features_path}"
            )

        self._model = joblib.load(self.model_path)
        self._features = joblib.load(self.features_path)

        logger.info(
            "Модель загружена: %s (%d признаков)",
            self.model_path.name,
            len(self._features),
        )

    @property
    def is_loaded(self) -> bool:
        """Проверяет, загружена ли модель."""
        return self._model is not None

    @property
    def features(self) -> list[str]:
        """Возвращает список признаков модели."""
        return list(self._features)

    def _prepare_dataframe(self, records: list[dict[str, float]]) -> pd.DataFrame:
        """Подготовка признаков и расчет доп. колонок."""
        df = pd.DataFrame(records)

        # Авто-расчет признаков из ноутбука, если их нет в запросе
        if "is_renovated" in self._features and "is_renovated" not in df.columns:
            if "yr_renovated" in df.columns:
                df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
        
        if "living_to_lot_ratio" in self._features and "living_to_lot_ratio" not in df.columns:
            if "sqft_living" in df.columns and "sqft_lot" in df.columns:
                df["living_to_lot_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)

        # Выравниваем порядок колонок ровно по обучающей выборке
        return df.reindex(columns=self._features, fill_value=0)

    def predict(self, data: dict[str, float]) -> float:
        """Предсказание цены для одного объекта."""
        if self._model is None:
            raise RuntimeError("Модель не загружена. Вызовите load() перед predict().")

        X = self._prepare_dataframe([data])
        prediction = self._model.predict(X)[0]

        # src/train.py сохраняет модель, которая сама восстанавливает цену.
        # Модель из ноутбука возвращает log1p(price) и требует обратного преобразования.
        price = prediction if isinstance(self._model, TransformedTargetRegressor) else np.expm1(prediction)
        return float(price)

    def predict_batch(self, records: list[dict[str, float]]) -> list[float]:
        """Пакетное предсказание стоимости."""
        if self._model is None:
            raise RuntimeError("Модель не загружена. Вызовите load() перед predict().")

        X = self._prepare_dataframe(records)
        predictions = self._model.predict(X)

        prices = predictions if isinstance(self._model, TransformedTargetRegressor) else np.expm1(predictions)
        return [float(price) for price in prices]


# Глобальный экземпляр для API
predictor = Predictor()
