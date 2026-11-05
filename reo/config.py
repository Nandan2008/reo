import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
WHISPER_MODEL = os.getenv("REO_WHISPER_MODEL", "base")
# Blank detects the language each time. Set e.g. "en" or "hi" to pin it and save a step.
LANGUAGE = os.getenv("REO_LANGUAGE", "").strip() or None

# "pyttsx3" is offline and instant; "orpheus" is expressive but needs the model
# terms accepted at console.groq.com and adds a second or two per reply.
TTS_BACKEND = os.getenv("REO_TTS", "pyttsx3")
TTS_MODEL = os.getenv("REO_TTS_MODEL", "canopylabs/orpheus-v1-english")
# Orpheus voices: autumn diana hannah austin daniel troy
TTS_VOICE = os.getenv("REO_TTS_VOICE", "autumn")
# Which installed Windows voice to use; matched against its name. Blank = system default.
VOICE_NAME = os.getenv("REO_VOICE", "zira")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = float(os.getenv("REO_SILENCE_THRESHOLD", "0.01"))
SILENCE_SECONDS = 1.2        # pause length that means "I'm done talking"
WAIT_FOR_SPEECH_SECONDS = 6  # give up if you say nothing
MAX_RECORD_SECONDS = 20

MAX_HISTORY_MESSAGES = 20

# How reo is summoned:
#   "hotkey" - a key combination toggles listening on and off
#   "wake"   - always listening, answers to its name
#   "push"   - press Enter in its window first
MODE = os.getenv("REO_MODE", "hotkey")
# pynput spelling: <ctrl> <alt> <shift> <cmd> plus a letter.
HOTKEY = os.getenv("REO_HOTKEY", "<ctrl>+<alt>+r")
# A small model is plenty for spotting one word, and keeps the idle cost down.
WAKE_MODEL = os.getenv("REO_WAKE_MODEL", "tiny")
# Whisper hears "reo" several ways, so accept the near misses.
_wake = [w.strip().lower() for w in
         os.getenv("REO_WAKE_WORDS", "reo,rio,rayo,reyo").split(",") if w.strip()]
WAKE_NAME = _wake[0] if _wake else "reo"   # what it calls itself
WAKE_WORDS = set(_wake)                     # what it answers to
# Accept near misses of the name. A weak mic turns "reo" into "rio", "reno", "leo".
WAKE_FUZZY = os.getenv("REO_WAKE_FUZZY", "1") not in ("0", "false", "no")
WAKE_SILENCE_SECONDS = 0.6   # shorter than a command pause: just end-of-sentence
WAKE_MAX_SECONDS = 8
