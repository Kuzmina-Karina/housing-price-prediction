"""
Вспомогательные функции для проекта прогнозирования стоимости недвижимости.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml


def setup_logging(logging_config: dict | None = None) -> None:
    """
    Настраивает базовое логирование.

    Параметры
    ----------
    logging_config : dict, optional
        Секция logging из config.yaml.
    """
    if logging_config is None:
        logging_config = {}

    level = getattr(logging, logging_config.get("level", "INFO").upper(), logging.INFO)
    fmt = logging_config.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    logging.basicConfig(level=level, format=fmt)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Загружает конфигурацию из YAML-файла.

    Параметры
    ----------
    config_path : str
        Путь к файлу конфигурации.

    Возвращает
    ----------
    dict
        Словарь с конфигурацией.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_dataset(config: dict) -> pd.DataFrame:
    """
    Загружает датасет King County House Sales.

    Сначала пытается загрузить через kagglehub.
    При ошибке использует fallback URL.

    Параметры
    ----------
    config : dict
        Конфигурация из config.yaml.

    Возвращает
    ----------
    pd.DataFrame
        Загруженный датафрейм.
    """
    data_config = config["data"]

    if data_config.get("source") == "kaggle":
        try:
            import kagglehub
            dataset_path = kagglehub.dataset_download(
                data_config["kaggle_dataset"]
            )
            df = pd.read_csv(f"{dataset_path}/kc_house_data.csv")
            logger = logging.getLogger(__name__)
            logger.info("Данные загружены через kagglehub")
            return df
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("kagglehub не сработал (%s). Использую fallback.", e)

    # Fallback
    url = data_config["fallback_url"]
    df = pd.read_csv(url)

    # Нормализация названий колонок
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    return df