#!/usr/bin/env bash
# ============================================================================
# test_request.sh — скрипт для тестирования API
# ============================================================================
# Использование:
#   bash scripts/test_request.sh              # health + predict
#   bash scripts/test_request.sh --health     # только health
#   bash scripts/test_request.sh --predict    # только predict
#   bash scripts/test_request.sh --batch      # пакетное предсказание
# ============================================================================

set -euo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"

# Тестовые данные (первая запись из датасета King County)
SAMPLE_PAYLOAD='{
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
    "sqft_lot15": 5650
}'

check_health() {
    echo "🔍 Health check: ${BASE_URL}/health"
    curl -s "${BASE_URL}/health" | python3 -m json.tool
    echo
}

predict() {
    echo "📊 Предсказание: ${BASE_URL}/predict"
    curl -s -X POST "${BASE_URL}/predict" \
        -H "Content-Type: application/json" \
        -d "${SAMPLE_PAYLOAD}" | python3 -m json.tool
    echo
}

predict_batch() {
    echo "📦 Пакетное предсказание: ${BASE_URL}/predict/batch"
    curl -s -X POST "${BASE_URL}/predict/batch" \
        -H "Content-Type: application/json" \
        -d "{\"records\": [${SAMPLE_PAYLOAD}, ${SAMPLE_PAYLOAD}]}" | python3 -m json.tool
    echo
}

# --- main ---

case "${1:-}" in
    --health)
        check_health
        ;;
    --predict)
        predict
        ;;
    --batch)
        predict_batch
        ;;
    *)
        echo "=========================================="
        echo "  Тестирование API прогнозирования"
        echo "  Базовый URL: ${BASE_URL}"
        echo "=========================================="
        echo
        check_health
        predict
        predict_batch
        ;;
esac