"""PII masking for emails and phone numbers (names are not masked: the assistant must say who owns what)."""
import re

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# 10-digit North American style (optional country code) and international numbers that start with "+".
# Needs 3-3-4 digit grouping, so dates (2026-03-12), amounts (148,409.38) and ids (TXN-00001) do not match.
PHONE = re.compile(
    r"(?<![\w.])\+\d{1,3}[\s.-]?\d{1,4}(?:[\s.-]?\d{2,4}){2,3}(?![\w])"
    r"|(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?![\w])"
)


def mask_pii(text: str) -> tuple[str, dict]:
    """Returns (masked_text, {"email": n, "phone": n})."""
    text, emails = EMAIL.subn("[EMAIL]", text)
    text, phones = PHONE.subn("[PHONE]", text)
    return text, {"email": emails, "phone": phones}


def count(masked: dict) -> int:
    return sum(masked.values())
