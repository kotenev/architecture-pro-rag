#! /usr/bin/env python
import os
import json
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from src.rag_bot import RAGBot
from src.config import FEWSHOT_FILE

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в переменных окружения")

bot = RAGBot()

fewshot_examples = []
if FEWSHOT_FILE and os.path.exists(FEWSHOT_FILE):
    with open(FEWSHOT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            fewshot_examples.append(json.loads(line))
    logger.info(f"Загружено {len(fewshot_examples)} Few-shot примеров")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = (
        "Привет! Я RAG-бот, посвящённый русской мифологии.\n\n"
        "Я могу отвечать на вопросы о духах, обрядах и легендах из нашей обновлённой базы знаний.\n\n"
        "Доступные команды:\n"
        "/start - показать это сообщение\n"
        "/help - получить помощь\n"
        "/stats - статистика базы знаний\n"
        "/fewshot_on - включить Few-shot примеры\n"
        "/fewshot_off - отключить Few-shot примеры\n\n"
        "Просто отправь вопрос о мифологических персонажах!"
    )
    await update.message.reply_text(welcome_msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Как мной пользоваться:\n\n"
        "1. Задай вопрос на русском языке\n"
        "2. Я найду релевантную информацию в базе\n"
        "3. Сформирую ответ с пошаговым рассуждением (Chain-of-Thought)\n\n"
        "Примеры вопросов:\n"
        "• Как Якуб заботится о любимой лошади хозяев?\n"
        "• Что происходит с теми, кто мешает Ярополку в полдень?\n"
        "• Чем знаменита пророческая песнь Авдея?\n\n"
        "Если я не найду информацию, честно скажу: 'Я не знаю'."
    )
    await update.message.reply_text(help_text)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    num_chunks = len(bot.chunks)
    stats_text = (
        f"Статистика базы знаний:\n\n"
        f"Количество чанков: {num_chunks}\n"
        f"Топ-K для поиска: {bot.top_k}\n"
        f"LLM модель: {bot.yandex_model}\n"
        f"Few-shot примеров: {len(fewshot_examples)}\n"
        f"Режим: {context.user_data.get('use_fewshot', True) and 'Few-shot включён' or 'Few-shot выключен'}"
    )
    await update.message.reply_text(stats_text)


async def fewshot_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['use_fewshot'] = True
    await update.message.reply_text("Few-shot примеры включены")


async def fewshot_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['use_fewshot'] = False
    await update.message.reply_text("Few-shot примеры выключены")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "unknown"

    logger.info(f"Запрос от @{username} (ID: {user_id}): {user_query}")

    await update.message.chat.send_action("typing")

    use_fewshot = context.user_data.get('use_fewshot', True)
    fs_examples = fewshot_examples if use_fewshot else None

    try:
        response = bot.answer(user_query, fewshot_examples=fs_examples)

        answer = response["answer"]
        sources = response["source"]
        explain = response["explain"]

        reply = f"{answer}\n\n"

        if sources:
            reply += f"Источники: {', '.join(set(sources))}\n"

        if explain and explain != "OK":
            reply += f"\n🔍 {explain}"

        await update.message.reply_text(reply)
        logger.info(f"Ответ отправлен пользователю @{username}")

    except Exception as e:
        error_msg = "Произошла ошибка при обработке запроса. Попробуйте позже."
        logger.error(f"Ошибка: {e}")
        await update.message.reply_text(error_msg)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}")
    if update and update.message:
        await update.message.reply_text(
            "Произошла непредвиденная ошибка. Попробуйте позже."
        )


def main():
    logger.info("Запуск Telegram-бота...")

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("fewshot_on", fewshot_on))
    application.add_handler(CommandHandler("fewshot_off", fewshot_off))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.add_error_handler(error_handler)

    logger.info("Бот запущен и готов к работе!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
