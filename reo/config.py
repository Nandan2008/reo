import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# Anything slower than this is a stall, not an answer.
REQUEST_TIMEOUT = float(os.getenv("REO_REQUEST_TIMEOUT", "25"))
# Room for an answer. 500 was fine while every reply was one spoken sentence,
# but a program is not one spoken sentence, and reo would stop in the middle
# of writing one. It is a ceiling, not a target: a reply that does not need
# the room does not spend it.
MAX_REPLY_TOKENS = int(os.getenv("REO_MAX_REPLY_TOKENS", "2600"))
# How long the tools in one reply may take between them. Nothing used to cap
# this: read_page allows 15s, a search allows 8s across four backends, and the
# model can ask for several rounds of them. One reply in his log took 18.5
# seconds with two tool calls in it, which from where he is sitting is reo
# having died. Past this reo stops fetching and answers with what it has.
TOOL_BUDGET = float(os.getenv("REO_TOOL_BUDGET", "18"))
# The model on this machine gets far longer, because the first question after
# a gap has to load five gigabytes into the graphics card: measured at 57.8s
# cold against 0.5s warm. Twenty five seconds would fail every cold start,
# and failing is worse than waiting when the alternative is no answer at all.
LOCAL_TIMEOUT = float(os.getenv("REO_LOCAL_TIMEOUT", "180"))
# How hard a reasoning model should think before a one-sentence answer.
# "low", "medium", "high", or blank to leave it to the model.
REASONING_EFFORT = os.getenv("REO_REASONING", "low").strip()
# Where speech becomes words. "hosted" sends the recording to Groq's
# whisper-large-v3-turbo, which is the biggest whisper there is and already
# on the key: 0.20s against the local model's 2.65s on the same clip, with
# none of its 0.66-to-2.4 second swing. "local" runs the small model on this
# machine instead, and nothing ever leaves the laptop.
#
# The local model is loaded either way. It is what answers when there is no
# network, when the key is spent, or when Groq says no.
HEARING = os.getenv("REO_HEARING", "hosted").strip().lower()
HOSTED_WHISPER = os.getenv("REO_HOSTED_WHISPER", "whisper-large-v3-turbo")
# Short on purpose: past this it is quicker to do it here than keep waiting.
HOSTED_HEARING_TIMEOUT = float(os.getenv("REO_HOSTED_HEARING_TIMEOUT", "10"))

# "small" hears an accent better, and on this machine takes about 7s against
# base's 2.3s on the same clip - measured across thread counts and both int8
# modes, none of which moved it. That is slower than the problem it fixes, so
# base stays and the vocabulary hint below does the work instead.
WHISPER_MODEL = os.getenv("REO_WHISPER_MODEL", "base")
# Wider search through the possible transcriptions. Measured at +0.06s, which
# is as close to free as accuracy gets.
BEAM_SIZE = int(os.getenv("REO_BEAM", "5"))
# Names Whisper will not guess but you say often. Proper nouns are most of
# what it gets wrong, and it cannot spell what it has never been told.
VOCABULARY = os.getenv(
    "REO_VOCABULARY",
    "Reo, Claude, Hyderabad, Telangana, Telugu, Nandan, Chrome, Brave, "
    "Spotify, WhatsApp, YouTube, Notepad, VS Code, MrBeast").strip()
# Languages worth considering when none is pinned. Left to pick from all 99,
# Whisper read this accent as Russian and Spanish and produced nonsense.
LANGUAGES = [c.strip().lower() for c in
             os.getenv("REO_LANGUAGES", "en,te,hi").split(",") if c.strip()]
# Blank means reo works the language out from the first thing it hears and
# then reuses it. Detection costs about as much as the transcription itself,
# so setting "en", "hi", "te" ... skips even that first time.
LANGUAGE = os.getenv("REO_LANGUAGE", "").strip() or None

# "pyttsx3" is offline and instant; "orpheus" is expressive but needs the model
# terms accepted at console.groq.com and adds a second or two per reply.
TTS_BACKEND = os.getenv("REO_TTS", "pyttsx3")
TTS_MODEL = os.getenv("REO_TTS_MODEL", "canopylabs/orpheus-v1-english")
# Orpheus voices: autumn diana hannah austin daniel troy
TTS_VOICE = os.getenv("REO_TTS_VOICE", "autumn")
# Which installed Windows voice to use; matched against its name. Blank = system default.
VOICE_NAME = os.getenv("REO_VOICE", "zira")
# Playback speed. Orpheus speaks slowly; 1.0 is its own pace.
SPEECH_SPEED = float(os.getenv("REO_SPEECH_SPEED", "1.15"))

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = float(os.getenv("REO_SILENCE_THRESHOLD", "0.01"))
# A recording whose loudest moment is below this never reaches Whisper, which
# does not return nothing for silence - it invents. Measured on this laptop:
# room 0.0012, ordinary speech peaking around 0.06, so this sits ten times
# over the room and well under anything actually said.
SPEECH_FLOOR = float(os.getenv("REO_SPEECH_FLOOR", "0.012"))
# Whisper's own opinion that a segment is not speech. Above this it is dropped.
NO_SPEECH_PROB = float(os.getenv("REO_NO_SPEECH_PROB", "0.6"))
# The pause that means "done talking". 0.5 was cutting sentences in half: a
# measured recording showed an ordinary mid-sentence pause lasting 0.8s, after
# which reo had already stopped listening and transcribed the first half alone.
# That is where "Abad" came from - the tail of Hyderabad, arriving by itself.
SILENCE_SECONDS = float(os.getenv("REO_SILENCE_SECONDS", "0.9"))
# When a key ends the recording, silence is only a safety net. Three seconds
# was over-cautious: it was set from one recording where he paused for two
# seconds in the middle of a sentence, and it meant three seconds of nothing
# at the end of every command that did not end with a second key press.
# A second and a half covers an ordinary thinking pause, and the transcription
# now happens *inside* the wait rather than after it, so the whole of what is
# left here is the wait itself.
HOTKEY_SILENCE_SECONDS = float(os.getenv("REO_HOTKEY_SILENCE_SECONDS", "1.5"))
# How long a pause before reo quietly starts transcribing what it has so far.
# If he carries on talking the attempt is binned and nothing is lost; if he
# has finished, the words are ready by the time the silence is up.
EARLY_GUESS_SECONDS = float(os.getenv("REO_EARLY_GUESS_SECONDS", "0.45"))
# How much of what came *before* the key press to keep. Opening the microphone
# takes half a second on this laptop, and that half second used to be cut off
# the front of the first word.
PRE_ROLL_SECONDS = float(os.getenv("REO_PRE_ROLL_SECONDS", "0.8"))
# Hold the microphone open between commands so that half second is paid once,
# at startup, instead of every time the key is pressed. Nothing is kept: the
# audio goes into a ring of PRE_ROLL_SECONDS that overwrites itself. Set
# REO_KEEP_MIC_OPEN=0 to open the device per recording instead.
KEEP_MIC_OPEN = os.getenv("REO_KEEP_MIC_OPEN", "1") not in ("0", "false", "no")
# How long to wait for the speech model before giving up on it. Recognising
# six seconds of speech measures 0.66s warm and 2.4s after the laptop has been
# idle, and one clip once took 853 seconds and never did it again. Whatever
# that was, reo should say it did not catch you rather than wait all day.
TRANSCRIBE_TIMEOUT = float(os.getenv("REO_TRANSCRIBE_TIMEOUT", "20"))
WAIT_FOR_SPEECH_SECONDS = 6  # give up if you say nothing
MAX_RECORD_SECONDS = 20

# Each message is resent every request, against an 8000 token/minute cap.
MAX_HISTORY_MESSAGES = 8

# How reo is summoned:
#   "hotkey" - a key combination toggles listening on and off
#   "wake"   - always listening, answers to its name
#   "push"   - press Enter in its window first
MODE = os.getenv("REO_MODE", "hotkey")
# The window: "web" is the HTML one (glow, gradients), "tk" the plain canvas
# fallback, "off" for console only.
UI = os.getenv("REO_UI", "web").lower()
if UI in ("1", "true", "yes"):
    UI = "web"
elif UI in ("0", "false", "no"):
    UI = "off"
# Let reo run commands on the machine. Off unless you turn it on: it is driven
# by speech that mishears, and by web pages that can be written to steer it.
ALLOW_SHELL = os.getenv("REO_ALLOW_SHELL", "0") not in ("0", "false", "no")
# Let reo move and click the mouse. It has no way to see the screen, so it can
# only click where it is told - coordinates you give it, or ones a tool
# reported back. It is told never to guess.
ALLOW_MOUSE = os.getenv("REO_ALLOW_MOUSE", "1") not in ("0", "false", "no")
# The one folder reo may write into. On the Desktop on purpose: a file he
# cannot find is the same as no file. Everything reo writes lands here, and
# a name that would land anywhere else is refused rather than tidied up.
WORKSPACE = os.getenv(
    "REO_WORKSPACE", str(Path.home() / "OneDrive" / "Desktop" / "reo-work"))
if not Path(WORKSPACE).parent.exists():      # no OneDrive on this machine
    WORKSPACE = str(Path.home() / "Desktop" / "reo-work")
# pynput spelling: <ctrl> <alt> <shift> <cmd> plus a letter.
HOTKEY = os.getenv("REO_HOTKEY", "<ctrl>+<alt>+r")
# Wake detection: "phrase" uses a model trained on the wake word itself and is
# far more reliable; "whisper" transcribes the room and looks for a name, which
# allows a custom name but needs a clean microphone.
WAKE_ENGINE = os.getenv("REO_WAKE_ENGINE", "phrase")
# openWakeWord ships: hey_jarvis, alexa, hey_mycroft, hey_rhasspy
WAKE_PHRASE = os.getenv("REO_WAKE_PHRASE", "hey_jarvis")
WAKE_THRESHOLD = float(os.getenv("REO_WAKE_THRESHOLD", "0.5"))
# Used only by the whisper engine: a small model keeps the idle cost down.
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
