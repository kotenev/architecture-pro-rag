#! /usr/bin/env python

import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.llms import YandexGPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Необходимо установить переменную окружения TELEGRAM_BOT_TOKEN")
if not YANDEX_API_KEY:
    raise ValueError("Необходимо установить переменную окружения YANDEX_API_KEY")
if not YANDEX_FOLDER_ID:
    raise ValueError("Необходимо установить переменную окружения YANDEX_FOLDER_ID")


VECTORSTORE_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
YANDEX_GPT_MODEL_NAME = "yandexgpt"
YANDEX_GPT_TEMPERATURE = 0.0

def initialize_rag_components():
    logger.info("Загрузка векторной базы знаний...")
    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings=None,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Индекс загружен из {VECTORSTORE_PATH}.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке векторной базы: {e}")
        raise

    logger.info("Инициализация YandexGPT LLM...")

    llm = YandexGPT(
        api_key=YANDEX_API_KEY,
        folder_id=YANDEX_FOLDER_ID,
        model=YANDEX_GPT_MODEL_NAME,
        temperature=YANDEX_GPT_TEMPERATURE,
        llm_name="ai"
    )

    few_shot_examples = """
    Пример 1:
    Контекст: Mark Murray - это лесной дух, охраняющий лес и следящий за тем, чтобы люди вели себя уважительно по отношению к природе. Он может изменять облик и часто появляется в виде старого дуба или лохматого зверя.
    Вопрос: Кто такой Mark Murray?
    Ответ: 1. Я ищу информацию о 'Mark Murray' в предоставленном контексте. 2. В одном из фрагментов говорится, что Mark Murray - это лесной дух, охраняющий лес и следящий за тем, чтобы люди вели себя уважительно по отношению к природе. 3. Следовательно, Mark Murray - это лесной дух, охраняющий лес.

    Пример 2:
    Контекст: Leslie Ortega - это дух воды, связанный с реками и озёрами. Она может быть как доброжелательной, так и опасной. Её часто изображают с длинными волосами, покрытыми водорослями.
    Вопрос: Расскажи о Leslie Ortega.
    Ответ: 1. Я ищу информацию о 'Leslie Ortega' в предоставленном контексте. 2. В одном из фрагментов говорится, что Leslie Ortega - это дух воды, связанный с реками и озёрами. 3. Следовательно, Leslie Ortega - это дух воды, связанный с реками и озёрами.
    """

    system_instruction = f"""
    Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги. Используй только информацию из предоставленного контекста.
    Если информации в контексте недостаточно для ответа, скажи: "Я не знаю."
    {few_shot_examples}
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("ai", system_instruction),
        ("human", "Контекст:\n{context}\n\nВопрос: {question}\nОтвет:")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # k - количество возвращаемых чанков

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    logger.info("Цепочка RAG инициализирована с YandexGPT.")
    return rag_chain

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет приветственное сообщение."""
    await update.message.reply_text("Привет! Я RAG-бот, обученный на уникальной базе знаний. Задай мне любой вопрос!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает текстовое сообщение от пользователя и возвращает ответ RAG-бота."""
    user_question = update.message.text
    logger.info(f"Получен вопрос от {update.effective_user.id}: {user_question}")

    global rag_chain
    if not rag_chain:
        await update.message.reply_text("Ошибка: RAG-цепочка не инициализирована. Проверьте логи.")
        logger.error("RAG chain is not initialized.")
        return

    try:
        response_text = await rag_chain.ainvoke(user_question)
        logger.info(f"Ответ сгенерирован для {update.effective_user.id}.")
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        response_text = "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте снова позже."

    await update.message.reply_text(response_text)

def main():
    """Запускает Telegram-бота."""
    global rag_chain
    rag_chain = initialize_rag_components()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Запуск Telegram-бота...")
    application.run_polling()

if __name__ == "__main__":
    main()
