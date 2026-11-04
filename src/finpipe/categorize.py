"""Merchant categorization: rules and corrections first, an LLM only for the long tail.

Resolution order for every distinct merchant (first match wins):

1. **corrections**  the user's own CSV (``merchant,category``). Always wins.
2. **rules**        deterministic knowledge passed in by the pipeline (e.g. "every row for this
                    merchant is a transfer"). Free and exact, so the LLM never sees these.
3. **cache**        a previous LLM answer, stored as JSONL. Re-running costs nothing.
4. **LLM**          batched, structured output, validated item by item.

Why merchants and not transactions: a year of transactions collapses to a few dozen merchants,
so the LLM bill is bounded by *distinct merchants* and goes to ~zero after the first run.

Failure handling, because LLM output is untrusted input:

* Every item is validated with Pydantic against the :class:`Category` enum. A hallucinated
  category never reaches the output; that single item is rejected and retried.
* Merchants missing from a response, or answered invalidly, are retried (``max_attempts``).
* Merchants still unresolved fall back to ``Other`` with ``needs_review`` set. Fallbacks are
  *not* cached, so the next run tries again.
* Low-confidence answers are kept (and cached, so they are not paid for twice) but flagged
  ``needs_review``; they land in the review queue for the user to confirm via corrections.
* Cache writes happen after every batch, so a crash midway keeps the work already paid for.
* Configuration problems (bad API key) raise :class:`LLMConfigError` immediately instead of
  silently filing everything under ``Other``.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from pydantic import BaseModel, Field, ValidationError

from .categories import CATEGORY_VALUES, Category

log = logging.getLogger(__name__)

UNKNOWN_MERCHANT = "Unknown"
DEFAULT_REVIEW_THRESHOLD = 0.7

SOURCE_CORRECTION = "correction"
SOURCE_RULE = "rule"
SOURCE_CACHE = "cache"
SOURCE_LLM = "llm"
SOURCE_FALLBACK = "fallback"


class LLMConfigError(RuntimeError):
    """The LLM cannot be used at all (missing SDK or key, rejected credentials). Never retried."""


class LLMClient(Protocol):
    """What the categorizer needs from an LLM backend."""

    model: str

    def categorize_batch(self, merchants: Sequence[str]) -> list[dict]:
        """Return raw, *unvalidated* ``{"merchant", "category", "confidence"}`` dicts."""


class LLMItem(BaseModel):
    """One validated LLM answer."""

    merchant: str = Field(min_length=1)
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class MerchantCategory:
    merchant: str
    category: str
    confidence: float
    source: str
    model: str | None
    needs_review: bool
    categorized_at: str | None


def _key(merchant: str) -> str:
    return " ".join(merchant.split()).casefold()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _chunks(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i: i + size])


# --------------------------------------------------------------------------- corrections


def load_corrections(path: str | Path | None) -> dict[str, Category]:
    """Read ``merchant,category`` rows. A missing file means no corrections."""
    if path is None or not Path(path).exists():
        return {}
    by_name = {c.value.casefold(): c for c in Category}
    out: dict[str, Category] = {}
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or {"merchant", "category"} - set(reader.fieldnames):
            raise ValueError(f"{path}: header must contain 'merchant' and 'category'")
        for line_no, row in enumerate(reader, start=2):
            merchant = (row.get("merchant") or "").strip()
            category = (row.get("category") or "").strip()
            if not merchant and not category:
                continue
            if category.casefold() not in by_name:
                raise ValueError(
                    f"{path}:{line_no}: unknown category {category!r}; "
                    f"expected one of {CATEGORY_VALUES}"
                )
            out[merchant] = by_name[category.casefold()]
    return out


# --------------------------------------------------------------------------- cache


class _Cache:
    """Append-only JSONL of LLM answers; later lines override earlier ones."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.records: dict[str, dict] = {}
        if self.path and self.path.exists():
            for line_no, line in enumerate(self.path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    Category(rec["category"])
                    float(rec["confidence"])
                    self.records[_key(rec["merchant"])] = rec
                except (ValueError, KeyError, TypeError):
                    log.warning("ignoring bad cache line %s:%d", self.path, line_no)

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def add(self, recs: list[dict]) -> None:
        for rec in recs:
            self.records[_key(rec["merchant"])] = rec
        if self.path and recs:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- categorizer


class Categorizer:
    def __init__(
            self,
            client: LLMClient,
            cache_path: str | Path | None = None,
            corrections: Mapping[str, Category] | None = None,
            rules: Mapping[str, Category] | None = None,
            batch_size: int = 25,
            max_attempts: int = 3,
            review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    ):
        if batch_size < 1 or max_attempts < 1:
            raise ValueError("batch_size and max_attempts must be >= 1")
        self.client = client
        self.corrections = {_key(k): Category(v) for k, v in (corrections or {}).items()}
        self.rules = {_key(k): Category(v) for k, v in (rules or {}).items()}
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.review_threshold = review_threshold
        self.cache = _Cache(cache_path)
        self.stats: Counter[str] = Counter()

    def categorize(self, merchants: Iterable[str | None]) -> list[MerchantCategory]:
        """One result per distinct merchant, in first-seen order."""
        ordered: dict[str, str] = {}
        for m in merchants:
            if m is None:
                continue
            k = _key(m)
            if k and k not in ordered:
                ordered[k] = m

        results: dict[str, MerchantCategory] = {}
        pending: list[str] = []
        for k, m in ordered.items():
            hit = self._resolve_locally(k, m)
            if hit is None:
                pending.append(m)
            else:
                results[k] = hit
                self.stats[hit.source] += 1

        if pending:
            results.update(self._ask_llm(pending))
        return [results[k] for k in ordered]

    # ------------------------------------------------------------------ local resolution

    def _resolve_locally(self, key: str, merchant: str) -> MerchantCategory | None:
        if key in self.corrections:
            return MerchantCategory(
                merchant, self.corrections[key].value, 1.0, SOURCE_CORRECTION, None, False, None
            )
        if key in self.rules:
            return MerchantCategory(
                merchant, self.rules[key].value, 1.0, SOURCE_RULE, None, False, None
            )
        if key == UNKNOWN_MERCHANT.casefold():
            return MerchantCategory(
                merchant, Category.OTHER.value, 1.0, SOURCE_RULE, None, False, None
            )
        rec = self.cache.get(key)
        if rec is not None:
            conf = float(rec["confidence"])
            return MerchantCategory(
                merchant, rec["category"], conf, SOURCE_CACHE, rec.get("model"),
                conf < self.review_threshold, rec.get("categorized_at"),
            )
        return None

    # ------------------------------------------------------------------ LLM path

    def _ask_llm(self, pending: list[str]) -> dict[str, MerchantCategory]:
        done: dict[str, MerchantCategory] = {}
        remaining = list(pending)
        for attempt in range(1, self.max_attempts + 1):
            if not remaining:
                break
            if attempt > 1:
                self.stats["retried_merchants"] += len(remaining)
            for batch in _chunks(remaining, self.batch_size):
                done.update(self._call(batch))
            remaining = [m for m in remaining if _key(m) not in done]

        for m in remaining:
            log.warning("could not categorize %r after %d attempts", m, self.max_attempts)
            done[_key(m)] = MerchantCategory(
                m, Category.OTHER.value, 0.0, SOURCE_FALLBACK, None, True, None
            )
            self.stats[SOURCE_FALLBACK] += 1
        return done

    def _call(self, batch: list[str]) -> dict[str, MerchantCategory]:
        self.stats["llm_calls"] += 1
        try:
            raw = self.client.categorize_batch(batch)
        except LLMConfigError:
            raise
        except Exception as exc:  # transient API failure: leave the batch for a retry
            log.warning("LLM call failed: %s", exc)
            self.stats["llm_errors"] += 1
            return {}

        requested = {_key(m): m for m in batch}
        accepted: dict[str, LLMItem] = {}
        for entry in raw if isinstance(raw, list) else []:
            try:
                item = LLMItem.model_validate(entry)
            except ValidationError:
                self.stats["invalid_items"] += 1
                continue
            k = _key(item.merchant)
            if k not in requested or k in accepted:
                self.stats["ignored_items"] += 1  # hallucinated or duplicate merchant
                continue
            accepted[k] = item

        stamp = _now()
        out: dict[str, MerchantCategory] = {}
        records: list[dict] = []
        for k, item in accepted.items():
            name = requested[k]
            records.append({
                "merchant": name, "category": item.category.value,
                "confidence": item.confidence, "model": self.client.model,
                "categorized_at": stamp,
            })
            out[k] = MerchantCategory(
                name, item.category.value, item.confidence, SOURCE_LLM, self.client.model,
                item.confidence < self.review_threshold, stamp,
            )
            self.stats[SOURCE_LLM] += 1
        self.cache.add(records)  # persist per batch so a later crash keeps paid-for work
        return out


# --------------------------------------------------------------------------- review queue


def write_review_queue(results: Iterable[MerchantCategory], path: str | Path) -> int:
    """Write merchants needing a human look (lowest confidence first). Returns the row count."""
    # rows = sorted((r for r in results if r.needs_review), key=lambda r: (r.confidence, r.merchant))
    rows = sorted((r for r in results if r.needs_review or r.category == "Other" or r.merchant == "Unknown"),
                  key=lambda r: (r.confidence, r.merchant))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["merchant", "suggested_category", "confidence", "source"])
        for r in rows:
            w.writerow([r.merchant, r.category, f"{r.confidence:.2f}", r.source])
    return len(rows)


def as_dict(result: MerchantCategory) -> dict:
    return asdict(result)
