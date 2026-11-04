"""Merchant-name normalization (rules first, LLM for the long tail).

Bank descriptions look like ``SQ *BLUE BOTTLE COFFEE 0423 DENVER CO``. This module turns
them into a stable, human-readable merchant key (``Blue Bottle Coffee``). Only the *distinct*
normalized merchants are sent to the LLM categorizer, so the cleaner this step is, the fewer
LLM calls (and dollars) a year of transactions costs.

These are deliberately simple heuristics: they will mangle some names. That is acceptable,
because the key only needs to be *consistent*, and the manual-corrections table plus the LLM
step handle whatever the rules cannot.
"""

from __future__ import annotations

import re

# Card-processor and bank prefixes that carry no merchant information.
_PREFIX_RE = re.compile(
    r"^(?:"
    # r"SQ\s*\*|TST\*\s*|PP\*|PAYPAL\s*\*|DD\s*\*|"
    r"SP\s|SPO\*|SQ\s*\*|TST\*\s*|PP\*|PAYPAL\s*\*|DD\s*\*|"
    r"POS DEBIT\s+|CHECKCARD\s+\d{4}\s+|"
    r"PURCHASE AUTHORIZED ON \d{2}/\d{2}\s+"
    r")",
)

# (pattern, canonical name). First match wins. Patterns run on the prefix-stripped,
# upper-cased description.
# _ALIASES: list[tuple[re.Pattern[str], str]] = [
_ALIASES = [
    (re.compile(p), name)
    for p, name in [
        (r"BLUE BOTTLE COFFEE", "Blue Bottle Coffee"),
        (r"STARBUCKS", "Starbucks"),
        (r"KING SOOPERS", "King Soopers"),
        (r"WHOLEFDS|WHOLE FOODS", "Whole Foods"),
        (r"SAFEWAY", "Safeway"),
        (r"AMZN|AMAZON", "Amazon"),
        (r"TARGET", "Target"),
        (r"UBER", "Uber"),
        (r"LYFT", "Lyft"),
        (r"SHELL", "Shell"),
        (r"CONOCO", "Conoco"),
        (r"NETFLIX", "Netflix"),
        (r"SPOTIFY", "Spotify"),
        (r"APPLE\.COM", "Apple"),
        (r"CHIPOTLE", "Chipotle"),
        (r"DOORDASH", "DoorDash"),
        (r"CVS", "CVS Pharmacy"),
        (r"AIRBNB", "Airbnb"),
        (r"DELTA AIR", "Delta Air Lines"),
        (r"\bAMC\b", "AMC Theatres"),
        (r"DECENTRIX", "Decentrix Payroll"),
        (r"HELIUM", "Helium Mobile"),
        (r"HEL", "Factor Meals"),
        (r"40 BOXES", "40 Boxes"),
        (r"STARLINK", "Starlink"),
        (r"ACORNS", "Acorns"),
        (r"THE K9", "The K9 Body Shop"),
        (r"710 10TH ST", "710 10th St Parking"),
        (r"BESTBUY", "Best Buy"),
        (r"PILOT", "Pilot Gas"),
        (r"HOTELCOM", "Hotel.com"),
        (r"19CWORKSHOP.COM", "19cworkshop.com"),
        (r"ROCKET MORTGAGE", "Rocket Mortgage"),
        (r"DEPT EDUCATION PPD", "Student Loans"),
        (r"WWW COSTCO COM", "Costco.com"),
        (r"IPHONE CITIZ", "Iphone Payment"),
    ]
]

_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV "
    "NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
)

# Cities that show up in descriptions. Longest first so multi-word names win.
_CITIES = (
    "CENTRAL CITY",
    "BLACK HAWK",
    "LITTLETON",
    "LAKEWOOD",
    "BOULDER",
    "DENVER",
    "ARVADA",
    "AURORA",
    "GOLDEN",
)


def _strip_location(text: str) -> str:
    parts = text.split()
    if len(parts) > 1 and parts[-1] in _STATES:
        parts = parts[:-1]
    out = " ".join(parts)
    for city in _CITIES:
        if out.endswith(" " + city):
            return out[: -(len(city) + 1)]
    return out


def normalize_merchant(raw: str | None) -> str:
    """Return a stable merchant key for a raw bank description."""
    if not raw:
        return "Unknown"

    text = " ".join(raw.upper().split())
    text = _PREFIX_RE.sub("", text).strip()

    for pattern, canonical in _ALIASES:
        if pattern.search(text):
            return canonical

    text = text.replace("*", " ")
    kept: list[str] = []
    for token in text.split():
        if any(ch.isdigit() for ch in token) or token.startswith("#"):
            break  # store numbers, reference ids, dates: everything after is noise
        if not any(ch.isalnum() for ch in token):
            continue  # stray punctuation such as "-"
        kept.append(token)

    cleaned = _strip_location(" ".join(kept)).strip()
    return cleaned.title() if cleaned else "Unknown"
