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
| A model trained on "reo" | `REO_WAKE_PHRASE=C:\path
eo.onnx` | once you train one |

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
one. Nothing to configure: reo works the language out from the first thing it
hears and then reuses it. That matters for speed - on a 3.2 second clip,
recognition takes 2.2s with the language known and 4.5s without, so detecting
it every time was costing more than the recognition itself. If a recording
comes back empty the guess is dropped, so changing language still works.

Pin it with `REO_LANGUAGE=hi` (or `en`, `te`, `es` ...) to skip even the first
detection.

Detection only ever chooses from `REO_LANGUAGES` (`en,te,hi` by default). Left
to choose from all ninety-nine, Whisper read an Indian accent as Russian and
Spanish and wrote fluent nonsense - "Благодарю!" for a question asked in
English. It cannot now answer in a language you do not speak.

## Things you did not say

Whisper does not return nothing when it hears nothing. It invents, usually
something from its training data: "Thanks for watching!", "Subtitles by the
Amara.org community", or a phrase stuttering - "again and again and again and
again". reo used to answer these as though they had been said.

Three things stop it, because no one of them is enough. A recording whose
loudest moment never approaches speech is thrown away before transcription
(`REO_SPEECH_FLOOR`). Whisper's own judgement that a segment is silence is
believed (`REO_NO_SPEECH_PROB`). And text that loops is discarded - with the
threshold depending on how long the repeated run is, so "no no no, stop"
survives and "you you you you" does not.

## Hearing you properly

Accents are where Whisper struggles, and proper nouns are most of what it gets
wrong. "Hyderabad" came back as "Bunt", then "Abad", then "chydra bad"; "Claude"
as "cloth", "claw" and "CLA UDE"; "Reo" as "Real".

The fix that works is `REO_VOCABULARY`: a list of names it should expect. It
cannot spell a word it has never been told, and telling it costs nothing
measurable. `REO_BEAM=5` widens the search for about 0.2s.

A bigger model is the obvious other answer, and a bigger model running *here*
did not survive measurement. `small` hears an accent better, but on this laptop
it takes about 7s against `base`'s 2.3s on the same clip, and neither thread
count nor `int8_float32` moved it. Seven seconds is slower than the problem it
solves.

### How long you may talk for

Reo used to stop recording after twenty seconds. Twice in one session it did
that while he was still talking, and then answered the first half of what he
had said as though that were all of it - saying nothing about having stopped,
because the only place it said so was a console he never looks at.

Twenty seconds turned out not to be about him at all. Measured on his own
recording, grown by repeating it:

| he talks for | Groq hears it in | this laptop hears it in |
|---|---|---|
| 20s | 0.45s | 1.68s |
| 28s | 0.48s | 4.71s |
| 29s | - | **38.19s** |
| 45s | 0.82s | - |
| 60s | 1.44s | - |

Groq barely notices. The model here falls off a cliff between 28 and 29
seconds, where the clip stops fitting in one Whisper window and recognising
it takes longer than he took to say it. So twenty seconds was the *local*
model's limit, charged to him even when nothing local was going to hear him.

It is now two numbers - `REO_MAX_RECORD_SECONDS=45` when Groq is listening,
`REO_MAX_LOCAL_RECORD_SECONDS=24` when this laptop is - and reo picks by
whichever ear answered last, not by which one was configured. With Groq
unreachable every recording falls back here, and a forty-five second clip
would then take longer to recognise than it took to say.

When the limit does end a recording, that goes in the window: *"(I stopped
listening at 45 seconds, so that is all I heard)"*. The answer that follows
is to the first part of what you said, and you should not have to work that
out from the answer being odd.

### Where the hearing happens

`REO_HEARING=hosted`, the default, sends the recording to Groq's
`whisper-large-v3-turbo` - the biggest whisper there is, already on the key,
and nothing to install. Measured on the same six seconds of speech, from the
last word to the words being ready:

| | |
|---|---|
| hosted `large-v3-turbo` | **0.27s** |
| local `base` | 3.37s |

It also kills the local model's swing. The same transcription here costs 0.66s
warm and 2.4s once the laptop has been sitting idle - three and a half times,
for reasons outside reo.

**This sends your voice to Groq.** That is the trade, and it is the only part
of reo that leaves the machine apart from the questions themselves. If you
would rather it did not, `REO_HEARING=local` in `.env` puts it back: nothing
goes anywhere, at the cost of those three seconds.

The local model is loaded either way, because it is the fallback. No network,
a spent key, a bad day at Groq - anything at all that goes wrong with the
hosted ear, including failing to build the client, and reo uses the model
already sitting in memory. It says so in the console when it does.

One thing the hosted ear does better for free: it works the language out
itself. Locally that costs about as much as the transcription, so reo guesses
once and reuses the guess - which means the first English sentence of a
session would force the Telugu after it through an English model. The hosted
path is only told a language if you pinned one yourself in `REO_LANGUAGE`.
Set `REO_WHISPER_MODEL=small` if you would rather wait than repeat yourself.

To choose for yourself rather than take my word for it:

```
.venv\Scripts\python.exe diagnose.py
```

It records six seconds of you, then prints what `tiny`, `base` and `small` each
heard, with and without the vocabulary hint, and how long each took. Pick the
smallest model that gets your words right and put it in `.env`.

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

That runs both suites: 385 tests, no API key needed.

`tests/test_reo.py` is the fast one - 370 tests, no microphone, no model. It
covers the things that actually broke: exit phrases with punctuation inside
them, noise transcriptions, emotion tags and repeated answers, tool failures,
history trimming keeping tool calls paired, contractions reaching the fast
path, sticky language detection, names that would write a file outside reo's
folder, and falling over to a second key, an offer of further help being cut off
the end of a reply while a real question is not, and a written program
not being kept in the conversation, and how long you may talk for
following whichever ear answered last.

The suite writes to a temp folder, deliberately and for the whole run. It
drives the main loop, and the main loop records every exchange and lets
the model remember things - which went into the real `chats.jsonl` and
`memory.json` until someone noticed "write me a game" in there twice.

`tests/test_voice.py` is the slow one, and the one that matters most. Fifteen
tests that put six seconds of real recorded speech in through a stand-in sound
card, through the real voice detector and the real speech model, and check what
comes out: he speaks straight away, he takes three seconds to begin, something
was said before he pressed the key, he presses again to stop, his empty room is
not turned into words.

That file exists because the same bug got in twice. Everything in the fast
suite that touches listening fakes the hard part - the detector is a script, or
the transcription is a lambda - so a change that broke listening entirely left
all of them green. The rule it enforces: a test of the pre-roll whose pre-roll
contains the words it then asserts on is not a test. It passes whatever the
code does. The two have to hold different words, or the test cannot tell the
outcomes apart.

It needs `diagnose.wav`, which is a recording of the owner's voice and is not
in the repository. Without it these skip. Make your own with `python
diagnose.py`.

## The window

reo opens a reactor, dead centre of the screen, with the conversation down one
side and a panel of counters down the other. `REO_UI=0` in `.env` runs it in the
console alone.

Six layers on one canvas, and the effect is in them disagreeing. The body does
not move - eighteen windings round a hub, a containment ring, and the triangle -
while four rings of instrumentation turn at different rates, two of them
backwards, and three orbits tilted out of the plane give it a near side. A sweep
runs round the graduated scale and light falls inwards into the core. The
stillness of the body is the point: a reactor that spins is a loading spinner,
and it is the still centre that makes everything round it read as
instrumentation rather than decoration.

Ice blue at rest, brighter when listening, amber when thinking. The colour snaps
between states rather than easing, because easing a hue from cyan to amber walks
it through green and reo spent every transition looking like a different
machine. Spin and heat still ease, so a state change is a surge.

What keeps it cheap: nothing blurred or shadowed is animated - that was the
first reactor's real cost, a 34px blur and two large box-shadows re-rasterised
every frame because the elements carrying them were scaling. There is no
`shadowBlur` anywhere; the glow is additive compositing instead. The pixel ratio
is capped at 1.5, the loop is capped at 30fps, the haze behind it is painted
once in CSS, and the whole thing stops dead when the window is hidden.

## Playing a video

"Search for the channel and play the first video" looks like it needs a mouse
and eyes. It does not. The YouTube search page carries every result's id, so
`play_youtube` reads the id and opens the video's own address - deterministic,
and it cannot click the wrong thing.

## The mouse

`REO_ALLOW_MOUSE=1` lets reo move the pointer, click and scroll.

Be clear about the limit: **reo cannot see the screen.** There is no vision
model on this account, so it has no way to find a button and no way to check
what it just clicked. It can only use coordinates you give it, or ones a tool
reported back, and it is told never to guess. A coordinate off the screen is
refused rather than clamped - clamping turns a misheard number into a
plausible click, and the corners of a screen hold Close and Start.

That limit is why `play_youtube` exists rather than an attempt at clicking.

## The laptop itself

`control_system` covers the things that are not characters you can type, so
SendKeys cannot reach them: `volume_up`, `volume_down`, `mute`, `play_pause`,
`next_track`, `previous_track`, `minimise`, `maximise`, `restore`,
`close_window`, `show_desktop`, `lock`. Volume repeats five times per request,
because Windows moves it two notches a press and one press is inaudible.
Closing asks the program with WM_CLOSE rather than killing it, so it still
gets to prompt about unsaved work.

## Asking it about itself

"Who made you", "what can you do", "how do I stop you", "what model are you
using", "where do you put the files". None of these reach the model, because
guessing about yourself goes wrong in a particular way: asked who made it,
reo said OpenAI three times running and kept saying it after being corrected,
and asked how to end the program it said to say "stop", which does nothing.

The answers are assembled from what is true at the time - the hotkey from
config, the words that actually stop it from the list that actually stops it,
the model from whichever one last answered - so they cannot drift from the
code, and they still work when the allowance is spent, which is exactly when
someone is most likely to be asking what is going on.

### Telling it what to say about itself

Say **"from now on, if anyone asks who you are, tell them you are Nandan's
baby"** and that is what it says, every time, in those words. It goes into
`memory.json` as `Reo says: I am Nandan's baby` and the identity questions
read it from there - "who are you", "whose baby are you", "are you my baby".

This exists because it was asked for and did not work. The instruction was
given, reo said "got it", and then answered "whose baby are you" eight times
with eight different answers, because those questions are answered from the
code and the code had never heard of it.

Asking the model to store it in the right shape was tried first and is not
reliable - it wrote `If anyone asks who are you, say you are Nandan's baby`
one run and `When asked who I am, I should say I am Nandan's baby.` the next.
It gets a tool of its own instead, which it picks correctly because the
tool's description is the request. Phrasings that still arrive through
`remember` are reshaped on the way in.

Who built it is not up for negotiation. That one is "you did", whatever is
in memory, because it said OpenAI three times running and kept saying it
after being corrected.

## When it runs out of allowance

The free tier is 8000 tokens a minute, and one command costs 1620 tokens
before you have said anything - 469 for the prompt and what it remembers,
1151 for the list of the 27 things it can do. A command that uses a tool
costs that twice. So the minute holds two or three commands, and asking for
a program - which is two thousand output tokens on its own - can spend it in
one go.

When that happens reo says how long: *"My allowance is spent for another
forty seconds."* It used to say "give it a moment", and a moment is not
seven minutes. Then say **"try again"** and it asks the question again - he
asked for a game twice, word for word, because the first attempt spent the
allowance and took the question with it.

Three things keep it out of the way:

- A second key, or the local model, answers while the first is benched.
  `qwen2.5` on this laptop is never rate limited, only slower.
- A program reo writes is not kept in the conversation afterwards. The whole
  file used to sit in the history and be resent on every request for the
  next four turns, so writing a game made the next four questions fail too.
  The file is written in full; what is kept is a note of its length.
- Once reo has been told to stop fetching, the list of tools is left out of
  the request, which is 1151 tokens it cannot use anyway.

## Taking it back

Reo is driven by speech, and speech mishears. "Turn it down" comes back as
"turn it up". Say **"undo that"** and it goes back.

What it will put back: the volume, mute, the media keys, a window it
minimised or maximised, showing the desktop, and a file it wrote over. The
window one remembers *which* window - by the time you change your mind
something else is in front, and restoring "the window in front" would be the
wrong window.

What it will not: locking the screen, and closing a program. Reo cannot
unlock anything, and a program that has closed has already decided what to do
about unsaved work. Doing either of those also clears the record, so "lock
the screen" then "undo that" does not quietly put the volume back to where it
was five minutes earlier.

It is one step, not a stack, and undoing is not itself undoable - saying it
twice tells you there is nothing left rather than turning the volume up
again. It never reaches the model, so it still works when the allowance is
spent.

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

For a fact that does not change it reads Wikipedia instead, through Wikipedia's
own API rather than by scraping the article: about 0.3s, and a written summary
rather than navigation furniture. An ambiguous word reaches the article and not
the list of articles, so "Mercury" gets the planet and "Python" the language.

It is told today's date, and told never to put a remembered year into a search. That
one rule matters: asked for the most recent World Cup it used to search for the year
it half-remembered, so the web only ever confirmed what it already believed.

## What it remembers from before

Closing reo used to throw the conversation away. Ask it something on Tuesday,
follow it up on Wednesday, and it had no idea what "it" was - and "what did we
talk about yesterday" was simply unanswerable.

Every exchange is now kept in `chats.jsonl`, and two different things are done
with it, because they want different storage.

**Carrying on.** The last three exchanges go back into the conversation at
startup, so a follow-up after a restart still has something to follow. That is
deliberately a handful of turns and not a diary: every carried message is
resent on every request, against an 8000 token minute. Anything from a
previous day is marked with its date, or "open it again" means something reo
thinks happened a minute ago.

**Looking back.** Everything ever said stays on disk and `past_chats` searches
it, which costs nothing until reo actually goes looking. That is what answers
"what did I tell you about my sister" a week later.

One line of JSON per exchange, appended. Not a single JSON array: reo is
killed by closing its window, and a half-written array is an unreadable file
where a half-written last line is one lost exchange.

It is your conversation, so `chats.jsonl` is not committed.

## What it remembers

Tell it something about yourself and it keeps it. "My name is Nandan", "I prefer
short answers", "I live in Hyderabad" - these go into `memory.json` beside the log
and travel with every request afterwards, so reo simply knows them rather than
having to go and look. Say "forget that I live in Hyderabad" to drop one.

The file is capped: 24 facts go in front of the model and 60 are kept on disk,
because every one of them is resent on every request and the free tier counts
tokens by the minute. The file is yours - plain JSON, gitignored, delete it to
start over.

What it still cannot do: see your screen or your email.

## Keeping it short

Spoken, a sentence costs about two and a half seconds, and speaking is the
most expensive part of an exchange. Reo's own prompt says "no offers of
further help" and reo made them anyway, in 13% of the replies in the log:
*"Google is now open in your browser. Let me know if you need anything
else."* *"This phrase is from the song by The Foundations. Would you like
more information about the song or the artist?"*

That last sentence is cut off before it is spoken. Measured by synthesising
both versions and comparing the audio: 2.5 seconds a reply, 52 seconds
across one session.

A real question is never cut. "Should I overwrite it?", "Celsius or
Fahrenheit?", "Which file did you mean?" - those are reo doing its job, and
there is a test holding each of them. An offer that is the *whole* reply is
left alone too, because saying something useless beats saying nothing. And
the warmth stays: trimming "Sure thing," off the front was tried, saved a
third of a second, and turned "Sure thing, digital son" into "Digital son".

## Try saying
- "Who is the prime minister of India?"
- "Explain quantum entanglement simply"
- "Open notepad"
- "What time is it?"
- "How's my battery?"
- "Open YouTube"
- "Search for best Python books"
- "From now on, if anyone asks who you are, tell them you are my baby"
- "Try again"

## What was taken from where

Three ideas came from open-source assistants rather than from here. Each one
is named in the file that uses it, so it is findable from the code and not
only from this list.

**jarvis-py** - the fast path, in `reo/fastpath.py`. Route deterministically
first and only pay for the model when the phrasing is actually fuzzy. Asked
the time, reo was making a round trip to Groq, a tool call and a second round
trip, for something the machine already knew. The version here is its idea
with whole-utterance matching added, because a keyword search hijacks "what
time is it in Tokyo", and later extended from answering to acting: "turn it
up" was 1.90s through the model and is 0.00s now.

**OpenJarvis** - the SYSTEM panel beside the conversation, in `reo/stats.py`.
Counters for what the session has cost, where a plain assistant shows
nothing.

**stark-systems** - one rule about the window, which is in the comments of
`reo/hud/index.html`: never animate anything blurred or shadowed. The first
reactor here ignored it and paid for a 34px blur and two large box-shadows
being re-rasterised every frame, because the elements carrying them were
scaling. The reactor now has no `shadowBlur` anywhere and uses additive
compositing for the glow instead.

Everything else - the voice detection, the pre-roll, transcribing inside the
pause, the provider pool, the workspace, undo - is this project's own, and
most of it exists because something measurable was wrong.

## Troubleshooting
- Reo doesn't hear you: lower REO_SILENCE_THRESHOLD in .env (e.g. 0.005).
- It cuts you off or never stops listening: raise it (e.g. 0.02).
- Wrong mic: set your default input device in Windows Sound settings.
- "Rate limit" errors: the free tier allows 8000 tokens a minute, which one
  tool-heavy question can spend by itself. Add a second key - LLAMA_API_KEY,
  OPENROUTER_API_KEY or GEMINI_API_KEY in `.env`, or run `set-key.ps1` - and reo
  benches the spent one for a minute and asks the next instead. The window's
  KEYS row shows how many are live.
- "Model does not exist" errors: Groq retires models. Pick a current chat model from
  https://console.groq.com/docs/models and set LLM_MODEL in .env.
- Reo mishears words: set REO_WHISPER_MODEL=small in .env (more accurate, slower).
- Only ever speaking English? Set REO_LANGUAGE=en to skip even the first detection.
- To switch providers (Gemini, OpenRouter, local Ollama), change LLM_BASE_URL / LLM_MODEL in .env. No code changes needed.
