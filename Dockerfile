FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY src/ ./src/
COPY models/ ./models/
COPY config.yaml .

# Порт для FastAPI
EXPOSE 8000

# Запуск через uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]