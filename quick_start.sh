#!/bin/bash

# ./quick_start.sh [telegram|api|repl|test]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}RAG-бот - Быстрый старт${NC}"
echo "========================================"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 не найден. Установите Python 3.11+${NC}"
    exit 1
fi

echo -e "${GREEN}Python найден: $(python3 --version)${NC}"

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Создаю виртуальное окружение...${NC}"
    python3 -m venv .venv
fi

source .venv/bin/activate

if [ ! -f ".venv/installed" ]; then
    echo -e "${YELLOW}Устанавливаю зависимости...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    touch .venv/installed
    echo -e "${GREEN}Зависимости установлены${NC}"
else
    echo -e "${GREEN}Зависимости уже установлены${NC}"
fi

if [ ! -f ".env" ]; then
    echo -e "${RED}Файл .env не найден${NC}"
    echo -e "${YELLOW}Создайте .env на основе .env.example:${NC}"
    echo "cp .env.example .env"
    echo "# Затем заполните необходимые переменные"
    exit 1
fi

echo -e "${GREEN}Конфигурация найдена (.env)${NC}"

export "$(cat .env | sed 's/#.*//g' | xargs)"

# Проверка индекса
if [ ! -d "index" ] || [ ! -f "index/faiss.index" ]; then
    echo -e "${RED}Индекс не найден${NC}"
    echo -e "${YELLOW}Запустите построение индекса:${NC}"
    echo "python src/build_index.py --kb-dir ./knowledge_base --index-dir ./index"
    exit 1
fi

echo -e "${GREEN}Индекс найден${NC}"

MODE=${1:-telegram}

case $MODE in
    telegram)
        echo -e "${GREEN}Запуск Telegram-бота...${NC}"
        if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
            echo -e "${RED}TELEGRAM_BOT_TOKEN не задан в .env${NC}"
            exit 1
        fi
        python src/telegram_bot.py
        ;;

    api)
        echo -e "${GREEN}Запуск REST API...${NC}"
        uvicorn api:app --host 0.0.0.0 --port 8000 --reload
        ;;

    repl)
        echo -e "${GREEN}Запуск консольного REPL...${NC}"
        python repl.py
        ;;

    test)
        echo -e "${GREEN}Запуск тестов...${NC}"
        python test_dialogs.py
        ;;

    docker)
        echo -e "${GREEN}Запуск через Docker Compose...${NC}"
        docker-compose up --build
        ;;

    *)
        echo -e "${RED}Неизвестный режим: $MODE${NC}"
        echo "Использование: ./quick_start.sh [telegram|api|repl|test|docker]"
        exit 1
        ;;
esac
