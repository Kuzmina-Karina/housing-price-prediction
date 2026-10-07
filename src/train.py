import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

# 1. Загрузка данных
data_path = "data/table-data/buildings_prices.csv"
if not os.path.exists(data_path):
    data_path = "../data/table-data/buildings_prices.csv"

df = pd.read_csv(data_path)

# 2. Целевая переменная и отбор числовых признаков
target_col = "price"

feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in feature_cols:
    feature_cols.remove(target_col)
if "id" in feature_cols:
    feature_cols.remove("id")

X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Фильтрация колонок через стандартный ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[("num", "passthrough", feature_cols)],
    remainder="drop"
)

# 4. Основной пайплайн
base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(random_state=42))
])

# 5. Обертка над целевой переменной (log1p при fit / expm1 при predict)
final_model = TransformedTargetRegressor(
    regressor=base_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

# 6. Обучение
final_model.fit(X, y)

# 7. Сохранение через joblib
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/model.pkl")
joblib.dump(feature_cols, "models/model_features.pkl")

print("Модель успешно обучена и сохранена без кастомных классов!")