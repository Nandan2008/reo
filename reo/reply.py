"""Cutting the end off a reply that has stopped saying anything.

Reo's own prompt says "no offers of further help", and reo ignores it. From
his log, in its own words:

    "This phrase is from the song ... Would you like more information
     about the song or the artist?"
    "I'm Nanan Sai's baby. How can I assist you today, Nanan Sai?"
    "It sounds like you might be quoting a song or a movie. Do you want
     to know who sings ...?"

Each of those is a sentence he has to sit through before he can speak. Measured
by synthesising both versions and comparing the audio's length: 2.5 seconds a
reply, and speaking is already the most expensive part of an exchange.

Asking the model more firmly was tried and is not reliable - it is the same
model that was already told not to. This is done after the fact instead, where
it either happens or does not.

The one thing this must not do is eat a real question. "Which file did you
mean?" and "Celsius or Fahrenheit?" are reo doing its job; they are not offers,
and nothing here matches them. An offer is reo volunteering more work -
"would you like me to", "shall I", "anything else" - or a closing pleasantry
with no content at all. Those, and only those.
"""
import re

# Sentence enders, keeping the punctuation with the sentence it ends.
SENTENCES = re.compile(r"(?<=[.!?])[\s ]+")

# Reo volunteering more work, or signing off. Anchored at the start of the
# sentence: "I asked whether you wanted me to stop" is not an offer.
#
# Deliberately narrow. "Do you want" alone would have eaten "Do you want
# Celsius or Fahrenheit?", so it has to be followed by reo doing something
# or fetching something. And no bare "shall I" or "should I": "Should I
# overwrite it?" is a question he needs to be asked, not a pleasantry.
OFFERS = (
    r"(?:so,? )?(?:would|do|did) you (?:like|want|need)"
    r" (?:me to|to know|to hear|to see|more|any|anything|another)",
    r"want me to\b",
    r"(?:is there )?anything (?:else|more)\b",
    r"let me know\b",
    r"feel free\b",
    r"how (?:can|may|else can) i (?:help|assist|be of)",
    r"how would you like to (?:proceed|continue|go)",
    r"(?:i )?hope (?:that|this|it) helps",
    r"(?:i am|i'm|im) (?:here|happy|glad) to (?:help|assist)",
    r"(?:happy|glad) to help\b",
    r"what(?:'s| is) next\b",
    r"(?:you can|if you)\b(?=.{0,60}\b(?:just ask|ask me|say the word|let me know|tell me))",
)
AN_OFFER = re.compile(r"^(?:%s)" % "|".join(OFFERS), re.IGNORECASE)

# Opening pleasantries - "Sure," "Of course," - were cut here too, and that
# was wrong twice over. Run across his real replies it turned "Sure thing,
# digital son" into "Digital son", and all of the time is at the end anyway:
# a leading "Sure," is a third of a second. The warmth is the point of it, so
# it stays.


def _bare(sentence: str) -> str:
    """The sentence with its punctuation and quoting off, for matching."""
    return re.sub(r"^[\s\"'“‘(—-]+", "", sentence).strip()


def tighten(text: str) -> str:
    """Drop a trailing offer of further help.

    Never returns nothing: a reply that is only an offer stays as it is, since
    saying something useless beats saying nothing at all.
    """
    if not text or not text.strip():
        return text
    parts = [p for p in SENTENCES.split(text.strip()) if p.strip()]
    # From the back, while there is something in front worth keeping.
    while len(parts) > 1 and AN_OFFER.match(_bare(parts[-1])):
        parts.pop()
    return " ".join(parts).strip() or text
