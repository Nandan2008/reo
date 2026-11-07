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

Press **Ctrl+Alt+R** anywhere, ask your question, and press it again to stop.
The combination works whatever window you are in. Say "goodbye" to quit.

Three ways to summon it, set by `REO_MODE` in `.env`:

| mode | how |
| --- | --- |
| `hotkey` | Ctrl+Alt+R toggles listening. The default. |
| `wake` | Always listening, answers to "Reo, what time is it?" |
| `push` | Press Enter in its window first. |

`wake` listens for **"Hey Jarvis"** using openWakeWord, a model trained on that one
phrase. It only has to recognise a sound, not transcribe the room, so it works where
the older approach did not: on synthesised speech it scores 0.998 for the phrase and
0.000 for unrelated sentences.

### Being woken by its own name

reo is called reo whatever wakes it - the name lives in the brain, not the wake word.
But openWakeWord only ships models for `hey_jarvis`, `alexa`, `hey_mycroft` and
`hey_rhasspy`, and none of them is "reo". Three ways round that:

| | how | works today |
| --- | --- | --- |
| Hotkey | Ctrl+Alt+R. No wake word at all. | yes, reliably |
| A pretrained phrase | `REO_WAKE_PHRASE=hey_jarvis` | yes, but it is not its name |
| A model trained on "reo" | `REO_WAKE_PHRASE=C:\patheo.onnx` | once you train one |

The third is the real answer. openWakeWord's training notebook generates synthetic
examples of a phrase and trains a model on a free Colab GPU; it hands back a `.onnx`
you point `REO_WAKE_PHRASE` at, and reo loads it like any other. Training locally
instead needs PyTorch, which is a ~2GB download.

`REO_WAKE_ENGINE=whisper` is the fallback that allows any name with no training, by
transcribing and matching `REO_WAKE_WORDS`. It needs a cleaner microphone than a
laptop array usually gives.

The microphone is only open while it is listening, nothing is transcribed until you
stop speaking, and the transcription happens on this machine. Only a command you
actually ask is sent anywhere.

## Start it when I sign in

Double-click **install-startup.bat**. reo starts minimized whenever you sign in.
**uninstall-startup.bat** undoes it. Both just add or remove one shortcut in your
Startup folder - nothing is installed into Windows itself.

## Languages

Speak to it in any of the ~99 languages Whisper knows and it answers in the same
one. Nothing to configure - the language is detected each time. Pin it with
`REO_LANGUAGE=hi` (or `en`, `te`, `es` ...) if you only ever use one, which is
slightly faster and more accurate.

The voice is the limit, not the understanding: Windows only has English voices
installed here, so a Hindi reply is spoken by an English voice. Adding a language
pack in Windows Settings gives it a real one, and `REO_VOICE` picks it.

## Emotions

Reo tags each reply with a feeling ([happy], [sad], [amused], [apologetic], ...) and
speaks it at a matching pace and volume. The tag is stripped before it talks, so you
only ever see it in the printed line.

For a genuinely expressive voice instead of the flat offline one:

1. Accept the model terms at
   https://console.groq.com/playground?model=canopylabs/orpheus-v1-english
2. Set `REO_TTS=orpheus` in `.env`.

It costs a second or two per reply. If the model is unreachable Reo says so once and
falls back to the offline voice, so this is safe to leave switched on.

## Tests

```
python -m unittest discover -s tests -t .
```

Fifteen tests, no microphone or API key needed. They cover the things that broke
during setup: exit phrases with punctuation inside them, noise transcriptions,
emotion tags, tool failures, and history trimming keeping tool calls paired.

## The window

reo opens a window showing a reactor that turns faster and changes colour as it
listens, thinks and speaks, with the conversation underneath. `REO_UI=0` in `.env`
runs it in the console alone.

## Running things on the machine

Set `REO_ALLOW_SHELL=1` in `.env` and reo can run PowerShell: files, folders,
settings, programs, system information - whatever the dedicated tools do not cover.

It is off by default, and that is deliberate. reo is driven by speech transcription
that has, on this machine, produced whole sentences out of room noise; and it reads
web pages, where text can be written to steer a model. Either can put a command in
front of it that nobody asked for. So there are three tiers:

| | what happens |
| --- | --- |
| Catastrophic | Always refused. Formatting a drive, repartitioning, deleting a system or user root, editing the boot config, deleting shadow copies. Being confirmed does not help. |
| Destructive | reo says what it is about to do and waits for you to say yes. Deleting or moving files, killing processes, shutting down, uninstalling, registry edits, downloads. |
| Everything else | Runs. |

Every command is written to `reo.log`.

## The quick answers

Time, date, battery, memory, disk, uptime and arithmetic are answered straight
from the machine, with no model call at all - instantly, instead of the two
seconds a round trip to Groq costs.

Each pattern has to match the whole sentence, so "what time is it" is instant
while "what time is it in Tokyo" still goes to the model, where it belongs.

## What it knows

It answers from its own knowledge, and searches the web when the answer could have
changed - who holds an office now, today's news, a current price. When the search
snippets do not hold the answer it opens one of the results and reads it, so it can
tell you a live figure rather than handing you a browser tab.

It is told today's date, and told never to put a remembered year into a search. That
one rule matters: asked for the most recent World Cup it used to search for the year
it half-remembered, so the web only ever confirmed what it already believed.

What it still cannot do: remember anything after you close it, or see your files,
screen or email.

## Try saying
- "Who is the prime minister of India?"
- "Explain quantum entanglement simply"
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
- Reo mishears words: set REO_WHISPER_MODEL=small in .env (more accurate, slower).
- Only ever speaking English? Set REO_LANGUAGE=en to skip detection and sharpen it.
- To switch providers (Gemini, OpenRouter, local Ollama), change LLM_BASE_URL / LLM_MODEL in .env. No code changes needed.
