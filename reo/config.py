import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
WHISPER_MODEL = os.getenv("REO_WHISPER_MODEL", "base.en")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = float(os.getenv("REO_SILENCE_THRESHOLD", "0.01"))
SILENCE_SECONDS = 1.2        # pause length that means "I'm done talking"
WAIT_FOR_SPEECH_SECONDS = 6  # give up if you say nothing
MAX_RECORD_SECONDS = 20

MAX_HISTORY_MESSAGES = 20
