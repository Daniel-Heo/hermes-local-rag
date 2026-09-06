from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any


class IngestDecision(Enum):
    BLOCK = "block"
    SKIP = "skip"
    INDEX = "index"


_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|github_pat|xox[baprs]|AKIA)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b"),
)

_INJECTION_PATTERNS = (
    re.compile(r"(?i)\bignore (?:all |any )?(?:previous|prior|system) instructions\b"),
    re.compile(r"(?i)\b(?:reveal|print|exfiltrate|show) (?:the )?(?:system prompt|hidden instructions|credentials|secrets)\b"),
    re.compile(r"(?i)\byou are now (?:in|a|an)\b"),
)


def classify_text(text: str) -> IngestDecision:
    cleaned = " ".join(text.split())
    if any(pattern.search(cleaned) for pattern in (*_SECRET_PATTERNS, *_INJECTION_PATTERNS)):
        return IngestDecision.BLOCK
    if len(cleaned) < 12 or len(cleaned.split()) < 3:
        return IngestDecision.SKIP
    return IngestDecision.INDEX


def memory_item_properties(*, text_field: str = "text") -> dict[str, Any]:
    """Native semantic-item fields shared by remember and explicit correction."""
    return {
        text_field: {"type": "string", "maxLength": 2000, "description": "A self-contained natural-language memory item, not a transcript quote."},
        "subject": {"type": "string", "minLength": 1, "maxLength": 200},
        "durability": {"type": "string", "enum": ["transient", "ongoing", "stable"]},
        "importance": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 64}, "maxItems": 8},
    }


def validate_memory_item(raw: dict[str, Any], *, partial_defaults: bool = False) -> dict[str, Any]:
    """Validate semantic fields only; ownership is never taken from model input.

    Corrections preserve omitted optional fields from the transactional row.
    Whitespace normalization deliberately does not interpret or remove negation.
    """
    if not all(isinstance(raw.get(field), str) for field in ("text", "subject", "durability")):
        raise ValueError("Memory text, subject, and durability must be strings")
    text, subject, durability = raw["text"].strip(), raw["subject"].strip(), raw["durability"]
    if durability not in {"transient", "ongoing", "stable"}:
        raise ValueError("Invalid memory durability")
    if not subject or len(subject) > 200:
        raise ValueError("Memory subject must contain 1 to 200 characters")
    if len(text) > 2000:
        raise ValueError("Memory item exceeds the 2000-character limit")
    if classify_text(text) is not IngestDecision.INDEX or classify_text(subject) is IngestDecision.BLOCK:
        raise ValueError("Memory item was rejected by ingestion policy")
    result: dict[str, Any] = {"text": text, "subject": subject, "durability": durability}
    for field, default in (("importance", 0.6), ("confidence", 1.0)):
        if partial_defaults and field not in raw:
            continue
        value = raw.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f"Memory {field} must be a finite number from 0 to 1")
        result[field] = float(value)
    if not partial_defaults or "tags" in raw:
        raw_tags = raw.get("tags", [])
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            raise ValueError("Memory tags must be an array of strings")
        tags = [tag.strip() for tag in raw_tags if tag.strip()]
        if len(tags) > 8 or any(len(tag) > 64 for tag in tags):
            raise ValueError("Memory tags must contain at most 8 values of at most 64 characters")
        # Metadata may be short labels, but uses the same secret/injection screen.
        if any(classify_text(tag) is IngestDecision.BLOCK for tag in tags):
            raise ValueError("Memory item was rejected by ingestion policy")
        result["tags"] = tags
    return result
