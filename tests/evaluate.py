import sys
import joblib
import yaml
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

MODEL_PATH = "models/model.pkl"
RANDOM_STATE = 42

# -------------------- Пороговые значения --------------------

REGRESSION_THRESHOLDS = {
    "r2": 0.50,
    "mae": 160000,
    "rmse": 250000,
}

TEXT_THRESHOLDS = {
    "f1": 0.75,
    "precision": 0.70,
    "recall": 0.70,
}

IMAGE_THRESHOLDS = {
    "accuracy": 0.80,
    "macro_f1": 0.75,
    "macro_precision": 0.75,
    "macro_recall": 0.75,
}


# ------------------------------------------------------------


def load_model():
    """Загрузка обученной модели."""
    return joblib.load(MODEL_PATH)


def table_data_func():
    """Задача A. Регрессия."""

    df = pd.read_csv("data/table-data/buildings_prices.csv")

    X = df.drop(columns=["price"])
    y = df["price"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = load_model()

    y_pred = model.predict(X_test)

    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
    }


def texts_func():
    """Задача B. Классификация текстов."""

    df = pd.read_csv("data/texts/review.csv")

    X = df["review"]
    y = df["sentiment"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = load_model()

    y_pred = model.predict(X_test)

    return {
        "f1": f1_score(y_test, y_pred, pos_label="positive"),
        "precision": precision_score(
            y_test,
            y_pred,
            pos_label="positive",
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            pos_label="positive",
        ),
    }


def images_func():
    """Задача C. Fashion-MNIST."""

    fashion = fetch_openml(
        name="Fashion-MNIST",
        version=1,
        as_frame=True,
    )

    X = fashion.data / 255.0
    y = fashion.target.astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = load_model()

    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(
            y_test,
            y_pred,
            average="macro",
        ),
        "macro_precision": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def check_thresholds(metrics, thresholds):
    """
    Проверка метрик относительно пороговых значений.
    """

    passed = True

    print("-" * 60)

    for metric, value in metrics.items():

        threshold = thresholds[metric]

        if metric in ("mae", "rmse"):
            ok = value <= threshold
            sign = "<="
        else:
            ok = value >= threshold
            sign = ">="

        status = "PASS" if ok else "FAIL"

        print(
            f"{metric:18}"
            f"{value:12.4f}   "
            f"(required {sign} {threshold})   "
            f"{status}"
        )

        passed &= ok

    print("-" * 60)

    return passed


def main():

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task = config["data_task_name"]["name"]

    if task == "table-data":

        print("Task A: Regression")

        metrics = table_data_func()
        thresholds = REGRESSION_THRESHOLDS

    elif task == "texts":

        print("Task B: Text Classification")

        metrics = texts_func()
        thresholds = TEXT_THRESHOLDS

    elif task == "images":

        print("Task C: Fashion-MNIST")

        metrics = images_func()
        thresholds = IMAGE_THRESHOLDS

    else:
        raise ValueError(f"Unknown task: {task}")

    passed = check_thresholds(metrics, thresholds)

    if passed:
        print("All metric thresholds passed.")
        sys.exit(0)

    print("Metric thresholds were not reached.")
    sys.exit(1)


if __name__ == "__main__":
    main()