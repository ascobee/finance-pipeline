"""Test doubles for the LLM backend: no network, no API key, no cost."""

from __future__ import annotations


class ScriptedClient:
    """Answers from a {merchant: (category, confidence)} map; can be scripted to misbehave.

    ``script`` is consumed one step per call. A step may be an Exception (raised), a callable
    taking the requested merchants, or a literal response list. When the script is empty the
    client answers normally. Unknown merchants get ("Other", 0.9).
    """

    model = "fake-model"

    def __init__(self, answers=None, script=None):
        self.answers = {k.casefold(): v for k, v in (answers or {}).items()}
        self.script = list(script or [])
        self.calls: list[list[str]] = []

    @property
    def requested(self) -> list[str]:
        return [m for call in self.calls for m in call]

    def entry(self, merchant):
        category, confidence = self.answers.get(merchant.casefold(), ("Other", 0.9))
        return {"merchant": merchant, "category": category, "confidence": confidence}

    def categorize_batch(self, merchants):
        self.calls.append(list(merchants))
        if self.script:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            if callable(step):
                return step(list(merchants))
            return step
        return [self.entry(m) for m in merchants]


class ScriptedSummaryClient:
    """Stands in for the summary writer. ``script`` steps: a string, an Exception (raised), or a
    callable taking the facts. With an empty script it returns a grounded template summary."""

    model = "fake-summary-model"

    def __init__(self, script=None):
        from finpipe.summary import template_summary

        self._template = template_summary
        self.script = list(script or [])
        self.feedback: list = []      # the feedback passed on each call
        self.calls = 0

    def write_summary(self, facts, feedback=None):
        self.calls += 1
        self.feedback.append(feedback)
        if not self.script:
            return self._template(facts)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(facts) if callable(step) else step
