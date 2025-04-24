import re
from typing import List, Dict, Tuple

# Ordered patterns: full_name first, then everything else
PII_PATTERNS: List[Tuple[str, str]] = [
    # Full name: 2–3 words, each starting with a capital and ≥3 letters
    ("full_name",     r"\b[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,}){1,2}\b"),
    # Email
    ("email",         r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    # Date of Birth (DD/MM/YYYY)
    ("dob",           r"\b\d{2}/\d{2}/\d{4}\b"),
    # Aadhar (exactly 12 digits)
    ("aadhar_num",    r"(?<!\d)\d{12}(?!\d)"),
    # Credit/Debit Card (13–19 digits, with optional spaces/dashes)
    ("credit_debit_no", r"\b(?:\d[ -]*?){13,19}\b"),
    # Card expiry (MM/YY or MM/YYYY)
    ("expiry_no",     r"\b(?:0[1-9]|1[0-2])/(?:\d{2}|\d{4})\b"),
    # Phone number: exactly 10 digits, optional +country prefix
    ("phone_number",  r"(?<!\d)(?:\+\d{1,3}[-\s]?)?\d{10}(?!\d)"),
    # CVV: standalone 3–4 digits
    ("cvv_no",        r"(?<!\d)\d{3,4}(?!\d)"),
]

def mask_pii(text: str) -> Tuple[str, List[Dict]]:
    """
    1. Scans text with PII_PATTERNS in order.
    2. Skips any match overlapping a higher-priority span.
    3. Records all entities.
    4. Replaces them in reverse order so spans don't shift.
    """
    entities: List[Dict] = []
    reserved_spans: List[Tuple[int,int]] = []

    for label, pattern in PII_PATTERNS:
        for m in re.finditer(pattern, text):
            start, end = m.span()
            # Skip if overlaps a span we’ve already reserved
            if any(not (end <= rs or start >= re) for rs, re in reserved_spans):
                continue
            reserved_spans.append((start, end))
            entities.append({
                "position": [start, end],
                "classification": label,
                "entity": m.group()
            })

    # Mask from end→start so earlier indices stay valid
    masked = text
    for ent in sorted(entities, key=lambda x: x["position"][0], reverse=True):
        s, e = ent["position"]
        masked = masked[:s] + f"[{ent['classification']}]" + masked[e:]

    return masked, entities
