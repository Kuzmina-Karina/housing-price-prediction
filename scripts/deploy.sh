#!/usr/bin/env bash
# ============================================================================
# deploy.sh — скрипт для сборки и запуска Docker-образа
# ============================================================================
# Использование:
#   bash scripts/deploy.sh              # сборка и запуск
#   bash scripts/deploy.sh --build-only # только сборка
#   bash scripts/deploy.sh --run-only   # только запуск
# ============================================================================

set -euo pipefail

IMAGE_NAME="housing-price-api"
PORT=${PORT:-8000}

build_image() {
    echo "🔨 Сборка Docker-образа: ${IMAGE_NAME}..."
    docker build -t "${IMAGE_NAME}" .
    echo "✅ Образ собран: ${IMAGE_NAME}"
}

run_container() {
    echo "🚀 Запуск контейнера на порту ${PORT}..."
    docker run -d \
        --name "${IMAGE_NAME}" \
        -p "${PORT}:8000" \
        --restart unless-stopped \
        "${IMAGE_NAME}"
    echo "✅ Контейнер запущен: http://localhost:${PORT}"
    echo "📄 Документация: http://localhost:${PORT}/docs"
}

stop_container() {
    echo "🛑 Остановка контейнера..."
    docker stop "${IMAGE_NAME}" 2>/dev/null || true
    docker rm "${IMAGE_NAME}" 2>/dev/null || true
    echo "✅ Контейнер остановлен"
}

# --- main ---

case "${1:-}" in
    --build-only)
        build_image
        ;;
    --run-only)
        run_container
        ;;
    --stop)
        stop_container
        ;;
    --restart)
        stop_container
        build_image
        run_container
        ;;
    *)
        build_image
        run_container
        ;;
esac