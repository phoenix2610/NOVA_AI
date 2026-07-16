#!/usr/bin/env python3
"""
NOVA LLM Router — Decides which brain model to use.

Primary  (llama3.2:1b)       — Fast, low-latency.  Operations, short queries, simple conversation.
Secondary (mistral:7b-instruct-q4_K_M) — Heavy, thorough.  Complex knowledge, code, unsure fallbacks.
"""

import re

# ── Model names (single source of truth) ─────────────────────────────────────
PRIMARY_MODEL   = "llama3.2:1b"
SECONDARY_MODEL = "mistral:7b-instruct-q4_K_M"

# ── Intents always routed to the fast primary model ───────────────────────────
# These only need a short Python snippet, not deep reasoning.
_PRIMARY_INTENTS = {
    "browser_op",
    "system_op",
    "media_op",
    "file_op",
}

# ── Keywords that signal complexity → secondary ───────────────────────────────
_COMPLEX_KEYWORDS = re.compile(
    r"\b(explain|describe|analyse|analyze|compare|difference|history|"
    r"how does|why does|summarise|summarize|essay|thesis|"
    r"write a|generate a|code|script|function|debug|error|bug|"
    r"philosophy|politics|economics|science|research|"
    r"latest|current|today|news|price|stock|weather)\b",
    re.IGNORECASE
)

# ── Query length threshold (chars) above which we escalate ────────────────────
_COMPLEX_CHAR_THRESHOLD = 80


def select_model(
    intent: str,
    query: str,
    has_web_context: bool = False,
) -> str:
    """
    Return the model name to use for this request.

    Logic (in priority order):
    1. Operation intents → Primary (1B is enough for xdg-open / wpctl snippets)
    2. Has web context   → Secondary (1B cannot reason well over long injected text)
    3. Long query        → Secondary
    4. Complex keywords  → Secondary
    5. Default           → Primary
    """

    # Operation: always fast path
    if intent in _PRIMARY_INTENTS:
        return PRIMARY_MODEL

    # Web context present: secondary to handle large injected text properly
    if has_web_context:
        return SECONDARY_MODEL

    # Long query: secondary
    if len(query) > _COMPLEX_CHAR_THRESHOLD:
        return SECONDARY_MODEL

    # Complex keyword: secondary
    if _COMPLEX_KEYWORDS.search(query):
        return SECONDARY_MODEL

    # Default: primary
    return PRIMARY_MODEL


def model_label(model: str) -> str:
    """Human-readable label for logging."""
    return "Primary-1B" if model == PRIMARY_MODEL else "Secondary-7B"
