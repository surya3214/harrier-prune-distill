from __future__ import annotations

import re


def normalize_text(text: str, *, normalize_whitespace: bool = True) -> str:
    if text is None:
        return ""
    cleaned = text.strip()
    if normalize_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def is_valid_text(text: str, min_chars: int) -> bool:
    return bool(text) and len(text) >= min_chars
