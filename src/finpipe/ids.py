"""Deterministic transaction identity.

A transaction ID is a hash of fields that never change between exports of the same
transaction. Reloading a file, or loading two exports whose date ranges overlap, therefore
produces the same IDs and the silver layer can dedupe on them.

Two decisions worth knowing about:

* The hash uses the *raw* description (whitespace-collapsed and upper-cased), not the
  normalized merchant. Normalization rules will keep improving; if they fed the ID, every
  improvement would silently re-key history.
* Money is integer cents, never floats, so formatting can never change a hash.

The Spark job builds the same string with built-in column functions (see silver.py) so the
hash can be computed in a distributed fashion without a Python UDF. A test asserts that the
two implementations agree. The ``re.ASCII`` flag keeps Python's ``\\s`` aligned with Java's.
"""

from __future__ import annotations

import hashlib
import re

_TRIM = re.compile(r"^\s+|\s+$", re.ASCII)
_WS = re.compile(r"\s+", re.ASCII)

ID_SEPARATOR = "|"


def description_key(raw: str) -> str:
    """Whitespace-collapsed, upper-cased description used as the stable identity text."""
    return _WS.sub(" ", _TRIM.sub("", raw)).upper()


def transaction_id(
    source: str,
    txn_date: str,
    amount_cents: int,
    desc_key: str,
    occurrence: int,
) -> str:
    """SHA-256 hex digest identifying one transaction.

    ``txn_date`` is an ISO ``YYYY-MM-DD`` string. ``occurrence`` is the 1-based index of
    this row among otherwise identical rows *within one source file* (two identical coffees
    on the same day get occurrence 1 and 2).
    """
    payload = ID_SEPARATOR.join(
        [source, txn_date, str(amount_cents), desc_key, str(occurrence)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
