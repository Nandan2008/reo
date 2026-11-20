import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import threading

from . import config, fastpath, providers
from .stats import session as stats
from .brain import Brain, OutOfAllowance
from .chats import chats
from .listen import MISSED, Listener
from .speak import said_aloud, speak
from . import tools
from .text import is_speech, normalize

# Whatever went unanswered - the allowance ran out, the network went - is
# kept, so he can say this instead of repeating himself.
# A bare "again", "say that again" and "do that again" are deliberately not
# here: those mean repeat what you just said, or do that action once more,
# and hijacking them would be worse than the problem.
AGAIN = {"try again", "try that again", "try it again", "ask that again",
         "ask it again", "retry", "have another go", "try once more"}

# Teaching reo whose voice to answer, and unteaching it. His words, from
# the log: "Remember my voice from now on, you should only respond to me."
LEARN_VOICE = {
    "remember my voice", "learn my voice", "remember my voice from now on",
    "only answer me", "only respond to me", "only listen to me",
    "only answer my voice", "learn my voice from now on",
    "you should only respond to me", "only talk to me",
}
FORGET_VOICE = {
    "forget my voice", "answer anyone", "listen to anyone",
    "respond to anyone", "stop recognising my voice",
    "stop recognizing my voice", "forget my voice from now on",
}

EXIT_PHRASES = {"goodbye", "bye", "exit", "quit", "stop listening", "goodbye reo", "bye reo"}

# Asked this, reo reports on the recording he has just made rather than on the
# microphone in the abstract: the real thing, at his real distance, with
# whatever the room was doing at the time. It never reaches the model.
HEARING_CHECK = {
    "can you hear me", "do you hear me", "microphone check", "mic check",
    "microphone test", "mic test", "sound check", "test the microphone",
    "how do i sound", "how am i coming through", "can you hear me ok",
    "can you hear me okay", "are you hearing me", "check the microphone",
}


# How long after reo stops talking its own voice might still come back. One
# turn of the loop he saw took about six seconds, so this is generous on
# purpose: the cost of being wrong is one ignored sentence, and the cost of
# being right is not holding a conversation with itself.
ECHO_WINDOW = 15.0


# A real word, said out loud, puts far more than this through the
# microphone - ordinary speech here peaks near 0.06 against a room's 0.0012.
# SPEECH_FLOOR sits between them, measured on his own laptop.
SURELY_SPOKE = 0.3          # seconds of voice behind an irreversible word


def _really_said(listener, text: str) -> bool:
    """Was there actually a voice behind that, or did whisper invent it?

    Only asked about ending the session, because that is the one thing that
    cannot be taken back.
    """
    from . import audio as au

    level = getattr(listener, "last_level", 0.0)
    voice = getattr(listener, "last_voice", 0.0)
    if level >= config.SPEECH_FLOOR and voice >= SURELY_SPOKE:
        return True
    # More than one word, and not one of whisper's stock answers, is nobody's
    # hallucination - "stop listening" is two words and a real instruction.
    return not au.from_silence(text) and len(normalize(text).split()) > 1


def _my_own_voice(text: str) -> bool:
    """Did the microphone just hand back what reo said a moment ago?

    He said "open youtube" and nothing else, and then watched this:

        REO  All set. Enjoy!
        YOU  Enjoy!
        REO  Thanks! You're welcome.
        YOU  You're welcome
        REO  Likewise.
        YOU  Likewise

    Every one of those is a piece of the line above it. The ring no longer
    keeps reo's own voice, which stops it at the source; this is the second
    line, for an echo that arrives late or comes back off a wall.
    """
    from . import speak as voice

    heard = normalize(text)
    said = normalize(voice.just_said())
    if not heard or not said:
        return False
    if time.time() - voice._went_quiet > ECHO_WINDOW:
        return False
    return heard in said


def _learn_his_voice(listener, ui, toggle, cut_in) -> None:
    """Record him a few times and keep what his voice sounds like.

    Three takes rather than one, and they are checked against each other
    before anything is stored - a voiceprint that cannot recognise its own
    owner would simply stop reo answering him.
    """
    from .voiceprint import TAKES, voiceprint

    _said(f"Say something ordinary, {TAKES} times, a few seconds each. "
          f"Go ahead.", ui, toggle, cut_in)
    takes = []
    for number in range(1, TAKES + 1):
        if ui:
            ui.set_state("listening")
        listener.listen(stop_now=(lambda: not toggle.listening)
                        if toggle is not None else None)
        clip = getattr(listener, "last_audio", None)
        voiced = getattr(listener, "last_voice", 0.0)
        if clip is None or voiced < 1.0:
            _said("I did not get enough of that one. Say a bit more.",
                  ui, toggle, cut_in)
            continue
        takes.append(clip)
        if number < TAKES:
            _said(f"Good. {TAKES - len(takes)} more.", ui, toggle, cut_in)
    ok, note = voiceprint.learn(takes)
    print(f"  [voice] {'enrolled' if ok else 'not enrolled'}: {note}")
    _said(note, ui, toggle, cut_in)


def _someone_else(listener) -> bool:
    """Was that somebody other than the person reo was taught?

    False whenever it cannot tell, which covers not being enrolled, a clip
    too short to judge, and the model being unavailable. Being unsure has to
    mean answering: reo going deaf is worse than reo answering the radio.
    """
    from .voiceprint import voiceprint

    if listener is None:
        return False
    return voiceprint.is_him(getattr(listener, "last_audio", None),
                             getattr(listener, "last_voice", 0.0)) is False


def _tee_to_logfile():
    """Started from the Startup folder there is no console to read, so keep a log.

    stderr goes in too. It did not, and that is why every crash in his log
    looks like a clean startup followed by nothing: a traceback goes to
    stderr, the window closes, and the log never heard about it.
    """
    path = Path(__file__).resolve().parent.parent / "reo.log"
    try:
        log = open(path, "a", encoding="utf-8", buffering=1)
    except OSError:
        return
    log.write(f"\n===== started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")

    class Tee:
        def __init__(self, also):
            self.also = also

        def write(self, text):
            if self.also is not None:
                self.also.write(text)
            log.write(text)
            return len(text)

        def flush(self):
            if self.also is not None:
                self.also.flush()
            log.flush()

    sys.stdout = Tee(sys.__stdout__)
    sys.stderr = Tee(sys.__stderr__)


def _log_crashes() -> None:
    """Write down anything that gets all the way out, from any thread.

    Without this a crash in the barge-in watcher or the microphone pump
    printed to a stderr nobody was reading and the thread died on the spot -
    which looks like reo going deaf or mute rather than like a bug.
    """
    def fell_over(kind, value, where, thread=None):
        who = f" in {thread}" if thread else ""
        print(f"\n  [crash]{who} {kind.__name__}: {value}")
        traceback.print_exception(kind, value, where)

    sys.excepthook = fell_over

    def in_a_thread(args):
        if args.exc_type is SystemExit:
            return
        fell_over(args.exc_type, args.exc_value, args.exc_traceback,
                  getattr(args.thread, "name", "a thread"))

    threading.excepthook = in_a_thread


def already_running() -> int:
    """PID of another reo, or 0. Two of them fight over the microphone and the
    speaker, and the loser waits forever on a sound device it cannot have."""
    import os

    try:
        import psutil
    except ImportError:
        return 0
    # The venv's python.exe launches a child with the same command line, so the
    # launcher looks exactly like a second copy. Skip the whole ancestor chain.
    family = {os.getpid()}
    try:
        walker = psutil.Process().parent()
        while walker is not None and len(family) < 8:
            family.add(walker.pid)
            walker = walker.parent()
    except Exception:
        pass

    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = process.info
            pid, name, cmdline = info["pid"], info["name"] or "", info["cmdline"] or []
        except Exception:
            continue
        if pid in family or not name.lower().startswith("python"):
            continue
        # "-m" immediately followed by "reo", and nothing looser. Matching any
        # part ending in "reo" caught the working directory and refused to start
        # when nothing was running at all.
        if any(a == "-m" and b == "reo" for a, b in zip(cmdline, cmdline[1:])):
            return pid
    return 0


def woken_by() -> str:
    """What to tell the user to say, which depends on the wake engine."""
    if config.WAKE_ENGINE == "phrase":
        from .phrase import pretty
        return pretty(config.WAKE_PHRASE)
    return config.WAKE_NAME


def _heard_detail(heard_in: float, listener) -> str:
    """Recording and transcribing in one line. They are both inside "heard",
    and only one of them is reo's fault."""
    stt = getattr(listener, "transcribe_seconds", 0.0)
    where = getattr(listener, "heard_by", "")
    if not stt:
        return f"heard {heard_in:.1f}s"
    return (f"heard {heard_in:.1f}s (you {max(heard_in - stt, 0.0):.1f}s + "
            f"stt {stt:.1f}s{' ' + where if where else ''})")


def _hand_back(listener) -> None:
    """Both sound devices are held open between commands, because opening
    them costs 373ms and 287ms and that was being paid every single time.
    Give them back on the way out."""
    from . import speak as voice
    for let_go in (listener.mic.close, voice.close):
        try:
            let_go()
        except Exception:
            pass


def _interrupt_watch(toggle):
    """Something speak() can ask whether it has been told to shut up.

    The key cannot be read as a flag here: in hotkey mode it stays on between
    commands, so "is it listening" is true the whole time. What matters is the
    key being pressed *again*, which the count gives us.
    """
    if toggle is None:
        return None
    pressed = toggle.presses
    return lambda: toggle.presses != pressed


def _said(reply, ui, toggle, cut_in) -> float:
    """Say it, and hand the floor straight back if he cut in.

    A reply to a misheard question used to have to be sat through to the end.
    Now a press stops it mid-word and starts recording, because interrupting
    means he wants to say something, not that he wants to be asked again.
    """
    started = time.time()
    cut_off = speak(reply, ui, until=cut_in)
    if cut_off:
        print("  (cut off - go on)")
        if ui:
            ui.line("note", "(cut off - go on)")
        if toggle is not None:
            toggle.listen_now()
    return time.time() - started


def handle(brain, text: str, ui=None, heard_in: float = 0.0, listener=None,
           toggle=None) -> bool:
    """Answer one command. Returns False when it is time to stop."""
    print(f"You: {text}")
    if ui:
        ui.line("you", text)
    # Said for longer than reo can listen for. The answer that follows is to
    # the first part of it, and he should not have to work that out from the
    # answer being odd.
    if listener is not None and getattr(listener, "cut_short", 0):
        note = (f"(I stopped listening at {listener.cut_short:.0f} seconds, "
                f"so that is all I heard)")
        print(f"  {note}")
        if ui:
            ui.line("note", note)
    # Taken now rather than when reo starts talking, so a press during the
    # thinking counts too: he wants to interject, and the reply he no longer
    # wants should not be read out first.
    cut_in = _interrupt_watch(toggle)

    # "Try again" rather than saying the whole thing over. He asked reo to
    # write him a game twice in a row, word for word, because the first
    # attempt ran the allowance out and the question went with it.
    # Nothing waiting falls through untouched: he may well mean "try opening
    # it again", which the model can work out from the conversation.
    if normalize(text) in AGAIN and getattr(brain, "unanswered", None):
        text = brain.unanswered
        print(f"  (asking again: {text})")

    if normalize(text) in HEARING_CHECK and listener is not None:
        spoke = _said(listener.how_i_hear_you(), ui, toggle, cut_in)
        stats.add_reply(heard_in, 0.0, spoke, used_model=False)
        print(f"  [time] {_heard_detail(heard_in, listener)}  "
              f"answered directly (no model)  spoke {spoke:.1f}s")
        return True

    if normalize(text) in LEARN_VOICE and listener is not None:
        _learn_his_voice(listener, ui, toggle, cut_in)
        return True

    if normalize(text) in FORGET_VOICE:
        from .voiceprint import voiceprint
        _said(voiceprint.forget(), ui, toggle, cut_in)
        return True

    if normalize(text) in EXIT_PHRASES:
        # The one irreversible thing reo does by voice, so it wants more than
        # a single word on trust. "Bye" arrived out of a silent room, from
        # whisper rather than from him, and ended a session he was using.
        if listener is not None and not _really_said(listener, text):
            note = "(I thought I heard goodbye, but there was nothing there)"
            print(f"  {note}")
            if ui:
                ui.line("note", note)
            return True
        speak("Goodbye.", ui)
        return False

    # Things the machine already knows, and things it can simply do, need no
    # model: answering the time took two seconds of round trips before this,
    # and "turn it up" took one and nine tenths. The window is told first,
    # because opening a program still takes a couple of seconds and it should
    # not look like nothing is happening.
    if ui:
        ui.set_state("thinking")
    quick = fastpath.answer(text)
    if quick is not None:
        spoke = _said(quick, ui, toggle, cut_in)
        # keep the exchange in history, or a follow-up loses the thread
        brain.history.append({"role": "user", "content": text})
        brain.history.append({"role": "assistant", "content": quick})
        chats.add(text, said_aloud(quick))
        stats.add_reply(heard_in, 0.0, spoke, used_model=False)
        print(f"  [time] {_heard_detail(heard_in, listener)}  "
              f"answered directly (no model)  spoke {spoke:.1f}s")
        return True

    started = time.time()
    try:
        reply = brain.think(text)
    except OutOfAllowance as e:
        # Not a fault: the free tier is used up for this minute. Writing code
        # spends output tokens fast, so this is the one he will actually meet.
        print(f"  [limit] {e}")
        how_long = getattr(e, "wait", "") or "a moment"
        brain.unanswered = text
        reply = (f"[apologetic] My allowance is spent for {how_long}. "
                 "Say try again and I will have another go.")
    except Exception as e:
        print(f"  [error] {e}")
        brain.unanswered = text
        reply = ("Something went wrong reaching my brain. "
                 "Say try again and I will have another go.")
    else:
        brain.unanswered = None
    thought = time.time() - started

    spoke = _said(reply or "Done.", ui, toggle, cut_in)
    # What was said out loud, not the string the model produced: the kept
    # conversation was holding "[neutral] She lives in Bangalore.", and that
    # tag then came back into the next session as something reo had said.
    chats.add(text, said_aloud(reply or "Done."))

    # Where a reply's time actually went, so slowness can be read off the log
    # instead of guessed at.
    stats.add_reply(heard_in, thought, spoke)
    calls = sum(1 for m in brain.history if m.get("role") == "tool")
    print(f"  [time] {_heard_detail(heard_in, listener)}  "
          f"thought {thought:.1f}s ({calls} tools)  "
          f"spoke {spoke:.1f}s  total {heard_in + thought + spoke:.1f}s")
    return True


def main():
    # Windows consoles are cp1252; replies often hold characters it cannot encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    _tee_to_logfile()
    _log_crashes()

    keys = [p for p in providers.build() if p.key != "your-groq-key-here"]
    if not keys:
        print("No model key. Copy .env.example to .env and add a free key, or run set-key.ps1.")
        return
    if len(keys) > 1:
        names = ", ".join(f"{p.name} ({p.label})" for p in keys)
        print(f"{len(keys)} keys: {names}")
        print("A key that hits its rate limit is benched for a minute and the next one answers.")

    other = already_running()
    if other:
        print(f"reo is already running (process {other}). Close that one first.")
        print("Two copies fight over the microphone and the speaker, which looks")
        print("exactly like reo being slow or frozen.")
        return

    # The first search of a session cost 5.3s against 1.4s for the next, all
    # of it building the client. Do it here, while the speech model loads.
    threading.Thread(target=tools.warm_up, daemon=True).start()

    listener, brain = Listener(), Brain()
    # Closing reo used to throw the conversation away, so a follow-up the
    # next morning had nothing to follow. The last few exchanges come back;
    # everything older is on disk and reo can search it when asked.
    earlier = chats.carry_over()
    if earlier:
        brain.history.extend(earlier)
        print(f"Carrying on from {len(earlier) // 2} earlier exchange(s); "
              f"{chats.how_many()} kept in all.")

    waker = toggle = combination = None
    if config.MODE == "wake":
        if config.WAKE_ENGINE == "phrase":
            from .phrase import PhraseListener
            waker = PhraseListener(listener)
        else:
            from .wake import WakeListener
            waker = WakeListener()
    elif config.MODE == "hotkey":
        from .hotkey import Toggle, describe
        combination = describe()
        toggle = Toggle(on_change=lambda on: print(
            f"\n[{combination}] {'listening' if on else 'stopped listening'}"))
        toggle.start()

    ui = None
    if config.UI != "off":
        if config.UI == "tk":
            from .ui import Display
        else:
            try:
                from .webui import Display
            except Exception as e:
                print(f"  [ui] the web window is unavailable ({e}); using the plain one.")
                from .ui import Display
        hint = {"hotkey": f"{combination} to listen",
                "wake": f'say "{woken_by()}"',
                "push": "press Enter in the console"}.get(config.MODE, "")
        ui = Display(subtitle=hint)
        try:
            ui.run(lambda: _loop(listener, brain, waker, toggle, combination, ui))
        finally:
            _hand_back(listener)
        return

    try:
        _loop(listener, brain, waker, toggle, combination, None)
    finally:
        _hand_back(listener)


def _loop(listener, brain, waker, toggle, combination, ui):
    speak("Reo online.", ui)
    why = "the window was closed"

    while not (ui and ui.closed):
        heard_in = 0.0
        try:
            if ui:
                ui.set_state("idle")
            if toggle:
                if not toggle.listening:
                    print(f"\n[Press {combination} to start listening]")
                    toggle.wait_until_listening()
                if ui:
                    ui.set_state("listening")
                started = time.time()
                # Pressing the key again ends the recording and answers what was
                # said. It used to throw the words away, which looked like reo
                # ignoring you entirely.
                text = listener.listen(stop_now=lambda: not toggle.listening)
                heard_in = time.time() - started
            elif waker:
                print(f'\n[Listening for "{woken_by()}"]')
                text = waker.wait()
                if not is_speech(text):
                    # woken with nothing after it, so ask what they want
                    speak("[happy] Yes?", ui)
                    if ui:
                        ui.set_state("listening")
                    text = listener.listen()
            else:
                input("\n[Press Enter, then speak]  ")
                print("Listening...")
                if ui:
                    ui.set_state("listening")
                text = listener.listen()
        except (KeyboardInterrupt, EOFError):
            why = "Ctrl+C"
            break
        except Exception as e:
            # A microphone that disappears, a driver that hiccups: say so and
            # go round again. Nothing here is worth ending the session over.
            print(f"  [error] could not listen: {type(e).__name__}: {e}")
            if ui:
                ui.line("note", "the microphone failed; press the key and try again")
            text = ""

        if _someone_else(listener):
            from .voiceprint import voiceprint
            note = (f"(that was not your voice, so I left it alone "
                    f"- {voiceprint.last_score:.2f})")
            print(f"  {note}")
            if ui:
                ui.line("note", note)
            continue

        if _my_own_voice(text):
            note = "(that was my own voice coming back - ignoring it)"
            print(f"  {note}")
            if ui:
                ui.line("note", note)
            continue

        if not is_speech(text):
            # A muted microphone, a silent room, a mumble and whisper
            # inventing words all printed the same line, and they need
            # completely different things done about them.
            note = MISSED.get(getattr(listener, "missed", ""), MISSED["unclear"])
            print(note)
            if ui:
                ui.line("note", note)
            continue

        try:
            keep_going = handle(brain, text, ui, heard_in, listener, toggle)
        except Exception as e:
            # handle() was outside this, so anything it raised - the voice,
            # the window, a tool with a surprise in it - ended the session
            # and took the traceback to an unread stderr with it.
            print(f"  [error] that one fell over: {type(e).__name__}: {e}")
            traceback.print_exc()
            if ui:
                ui.line("note", "something went wrong answering that - still here")
            keep_going = True
        if not keep_going:
            why = "you said goodbye"
            break
    else:
        why = "the window was closed"

    # The log used to simply stop, which told him nothing about whether reo
    # had been closed or had fallen over.
    print(f"\n===== reo stopped: {why} =====")


if __name__ == "__main__":
    main()
