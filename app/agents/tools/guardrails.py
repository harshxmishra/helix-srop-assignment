"""
E5 Guardrails: refusal on out-of-scope, PII redaction.
"""
import re


OUT_OF_SCOPE_KEYWORDS = [
    "write me a poem",
    "tell me a joke",
    "how to hack",
    "how to cheat",
    "romance",
    "recipe",
    "sports scores",
    "political opinion",
    "medical advice",
]

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
}


def is_out_of_scope(query: str) -> bool:
    """Check if query is out of scope for support agent."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in OUT_OF_SCOPE_KEYWORDS)


def redact_pii(text: str) -> str:
    """Replace PII with placeholders."""
    result = text
    replacements = {
        "email": "[EMAIL]",
        "phone": "[PHONE]",
        "ssn": "[SSN]",
        "credit_card": "[CARD]",
    }
    for pii_type, pattern in PII_PATTERNS.items():
        result = re.sub(pattern, replacements[pii_type], result, flags=re.IGNORECASE)
    return result
