import os
from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

# Чтение токена из переменной окружения
TOKEN = os.getenv("8103299176:AAFV78XeN7llI2rX6thy4_BrWvT2wTTGSGE")

if not TOKEN:
    raise ValueError("Переменная окружения BOT_TOKEN не установлена!")