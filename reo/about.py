"""What reo knows about itself.

Asked these, reo used to guess, and guessing about yourself goes wrong in a
particular way: it said it was made by OpenAI three times running and kept
saying it after being corrected, and it told him to say "stop" to end the
program, which is not one of the words that works.

None of this needs a model. The name is in config, the words that stop it
are in __main__, the tools are a list, the folder is a path, and which model
is answering is whatever last answered. So these are assembled from what is
actually true at the time and never leave the machine - which also means
they still work when the allowance is spent, which is exactly when someone
is most likely to be asking what is going on.
"""
from . import config


def _stopping() -> str:
    from .__main__ import EXIT_PHRASES
    words = sorted(p for p in EXIT_PHRASES if " " not in p)
    return (f"Say {', '.join(words[:-1])} or {words[-1]}. "
            "Ctrl+Q closes the window, and Escape takes it out of full screen.")


def _who_built_it() -> str:
    return ("You did. I run entirely on this laptop - the code is in your reo "
            "folder, and you can change any of it.")


def _told_to_say() -> str:
    """Whatever he has told reo to say about itself. Empty if he has not."""
    from .memory import memory
    said = [s.rstrip(".") for s in memory.about_reo() if s.strip()]
    return " ".join(s + "." for s in said)


def _told_only():
    """For questions only he can have given reo an answer to - "whose baby
    are you" and the like. None when he has not, so the model gets it."""
    return _told_to_say() or None


def _what_it_is() -> str:
    from .hotkey import describe
    how = {"hotkey": f"press {describe()}",
           "wake": "say the wake word",
           "push": "press Enter in my window"}.get(config.MODE, "press the key")
    # What he told reo to say comes first, because he asked for it and the
    # rest he already knows.
    return (f"{_told_to_say()} I am Reo, a voice assistant running on this "
            f"laptop. To talk to me, {how}.").strip()


def _what_it_can_do() -> str:
    return ("I can open programs and websites, play a video, search the web "
            "and read a page, write files and run them, control the volume, "
            "the music and the windows, move the mouse where you tell me, "
            "remember things about you, and put the last thing back if I got "
            "it wrong. Say undo that.")


def _which_model() -> str:
    from . import providers
    thinking = providers.active_label()
    ear = ("Groq's big whisper" if config.HEARING == "hosted"
           else f"the {config.WHISPER_MODEL} speech model here")
    return f"I think with {thinking}, and I hear you with {ear}."


def _whose_voice() -> str:
    """Whether reo has been taught a voice, and how well it took.

    Worth asking out loud: the number from enrolment is the only thing that
    says whether this works in his room, and he is the only one who can hear
    the answer.
    """
    import json

    from .voiceprint import voiceprint

    if not voiceprint.enrolled:
        return ("I answer anyone. Say remember my voice and I will learn "
                "yours.")
    try:
        kept = json.loads(voiceprint.path.read_text(encoding="utf-8"))
        agreement = float(kept.get("agreement", 0))
        when = kept.get("since", "")
    except (OSError, ValueError, TypeError):
        return "I know your voice, and I only answer you."
    return (f"I know your voice - learnt {when}, and your recordings matched "
            f"at {agreement:.2f} out of one. I only answer you. Say forget "
            f"my voice to undo that.")


def _where_files_go() -> str:
    return f"Everything I write goes in {config.WORKSPACE}."


def _what_it_hears_with() -> str:
    return _which_model()


# Whole-utterance patterns, the same rule as everything else in fastpath.
ABOUT = [
    (r"(who|what) (made|created|built|wrote|designed) you"
     r"|who( is|s) your (maker|creator|developer)"
     r"|(who|what) are you (made|created|built) by", _who_built_it),
    (r"whose \w+ (are|r) you|whos \w+ are you|are you (my|mine)( \w+)?", _told_only),
    (r"(who|what) are you|what are you called|whats your name"
     r"|introduce yourself|tell me about yourself", _what_it_is),
    (r"(what|which) (can|could) you do|what are you (able to do|capable of)"
     r"|what can you help( me)? with|what do you do"
     r"|(what|which) things can you do", _what_it_can_do),
    (r"how do i (stop|close|quit|exit|end) (you|this|the (program|app))"
     r"|how do i turn you off|how do i shut you (down|up)"
     r"|how do i end the program|how do i get rid of you", _stopping),
    (r"(what|which) (model|brain|ai) (are you|do you) (using|use|run on|running)"
     r"|what are you (built|running) on|what model are you"
     r"|(what|which) model is (this|that)", _which_model),
    (r"(do you )?(know|remember|recognise|recognize) my voice"
     r"|whose voice (do you|are you) (know|answer|answering|listening to)"
     r"|do you only (answer|respond to) me"
     r"|have you learn(t|ed) my voice", _whose_voice),
    (r"where do you (put|save|write|keep) (the |my |your )?(files|code|things|stuff)"
     r"|where (do|does) (the |my |your )?(files|code) go"
     r"|wheres your folder|where is your folder", _where_files_go),
]
