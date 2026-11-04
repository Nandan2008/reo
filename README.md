# reo

Voice assistant for Windows. Day 1: push-to-talk voice loop with a free open-source LLM (gpt-oss on Groq) + basic tools.

## Quick start (easiest)

1. Double-click **setup.bat**. It installs everything, opens the Groq page so you can grab a free key, and starts reo.
2. Next time, just double-click **run.bat**.

## Manual setup (Windows, PowerShell)

1. Install Python 3.11 or newer from python.org (tick "Add to PATH"). Tested on 3.14.
2. In this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env
   ```
3. Get a free API key at https://console.groq.com (no card needed), paste it into `.env` as LLM_API_KEY.
4. Run:
   ```
   python -m reo
   ```

Press Enter, speak, pause. Say "goodbye" to exit.

## Try saying
- "Open notepad"
- "What time is it?"
- "How's my battery?"
- "Open YouTube"
- "Search for best Python books"

## Troubleshooting
- Reo doesn't hear you: lower REO_SILENCE_THRESHOLD in .env (e.g. 0.005).
- It cuts you off or never stops listening: raise it (e.g. 0.02).
- Wrong mic: set your default input device in Windows Sound settings.
- "Rate limit" errors: you hit the free tier's per-minute limit. Wait a minute, or switch LLM_MODEL to a smaller model.
- "Model does not exist" errors: Groq retires models. Pick a current chat model from
  https://console.groq.com/docs/models and set LLM_MODEL in .env.
- Reo mishears words: set REO_WHISPER_MODEL=small.en in .env (more accurate, slower).
- To switch providers (Gemini, OpenRouter, local Ollama), change LLM_BASE_URL / LLM_MODEL in .env. No code changes needed.
