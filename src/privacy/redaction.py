from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: Dict[str, int]


def _hash_token(value: str) -> str:
    """Stable short hash for redacted identifiers (debuggable without leaking raw values)."""
    h = sha256(value.encode("utf-8")).hexdigest()
    return h[:10]


def redact_text(text: str, *, extra_terms: Optional[Iterable[str]] = None) -> RedactionResult:
    """
    Best-effort PII redaction.

    This is intentionally conservative: it targets common identifiers (emails, phones, MRNs, DOBs,
    SSNs, and common "Patient Name:" style fields). It cannot perfectly detect all names/PII.
    """
    if not text:
        return RedactionResult(text="", counts={})

    counts: Dict[str, int] = {}
    out = text

    def bump(label: str, n: int = 1) -> None:
        counts[label] = counts.get(label, 0) + n

    # Emails
    email_re = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

    def _email_sub(m: re.Match) -> str:
        bump("email")
        return f"[REDACTED_EMAIL:{_hash_token(m.group(0))}]"

    out = email_re.sub(_email_sub, out)

    # Phone numbers (US-ish / international-ish)
    phone_re = re.compile(r"(?<!\d)(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")

    def _phone_sub(m: re.Match) -> str:
        bump("phone")
        return f"[REDACTED_PHONE:{_hash_token(m.group(0))}]"

    out = phone_re.sub(_phone_sub, out)

    # SSN
    ssn_re = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    def _ssn_sub(m: re.Match) -> str:
        bump("ssn")
        return "[REDACTED_SSN]"

    out = ssn_re.sub(_ssn_sub, out)

    # Dates of birth (common formats)
    dob_re = re.compile(
        r"(?i)\b(DOB|Date\s*of\s*Birth)\b\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}|[0-9]{4}-[0-9]{2}-[0-9]{2})"
    )

    def _dob_sub(m: re.Match) -> str:
        bump("dob")
        return f"{m.group(1)}: [REDACTED_DOB]"

    out = dob_re.sub(_dob_sub, out)

    # Standalone date formats (can be over-inclusive; still useful for medical PDFs)
    date_re = re.compile(r"\b([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b")

    def _date_sub(m: re.Match) -> str:
        bump("date")
        return "[REDACTED_DATE]"

    out = date_re.sub(_date_sub, out)

    # Medical record numbers / patient IDs / account numbers (label-based)
    mrn_re = re.compile(
        r"(?i)\b(MRN|Medical\s*Record\s*Number|Patient\s*ID|Account\s*(No|Number)|Acc\s*#)\b\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-]{3,})"
    )

    def _mrn_sub(m: re.Match) -> str:
        bump("mrn")
        return f"{m.group(1)}: [REDACTED_ID:{_hash_token(m.group(3))}]"

    out = mrn_re.sub(_mrn_sub, out)

    # Patient name fields (label-based; avoids trying to solve generic NER)
    name_re = re.compile(r"(?i)\b(Patient\s*Name|Name)\b\s*[:\-]?\s*([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4})")

    def _name_sub(m: re.Match) -> str:
        bump("name")
        return f"{m.group(1)}: [REDACTED_NAME:{_hash_token(m.group(2))}]"

    out = name_re.sub(_name_sub, out)

    # Addresses (label-based, best-effort)
    addr_re = re.compile(r"(?i)\b(Address)\b\s*[:\-]?\s*([^\n]{8,120})")

    def _addr_sub(m: re.Match) -> str:
        bump("address")
        return f"{m.group(1)}: [REDACTED_ADDRESS]"

    out = addr_re.sub(_addr_sub, out)

    # User-provided extra terms (names, clinics, etc.)
    if extra_terms:
        for term in extra_terms:
            t = (term or "").strip()
            if not t:
                continue
            # Whole-word match if it looks word-ish, otherwise plain escape.
            pattern = r"\b" + re.escape(t) + r"\b" if re.match(r"^[A-Za-z0-9_ -]+$", t) else re.escape(t)
            term_re = re.compile(pattern, re.IGNORECASE)
            if term_re.search(out):
                out, n = term_re.subn("[REDACTED_CUSTOM]", out)
                if n:
                    bump("custom", n)

    return RedactionResult(text=out, counts=counts)


def redact_sources(sources: List[dict], *, extra_terms: Optional[Iterable[str]] = None) -> List[dict]:
    """
    Redact `chunk` fields in retrieval sources (in-place copy).
    """
    out: List[dict] = []
    for s in sources or []:
        d = dict(s)
        chunk = d.get("chunk")
        if isinstance(chunk, str) and chunk:
            d["chunk"] = redact_text(chunk, extra_terms=extra_terms).text
        out.append(d)
    return out


def sanitize_filename(filename: str) -> str:
    """
    Avoid persisting potentially identifying filenames.
    Keeps extension for UX, but replaces base with a stable hash.
    """
    name = (filename or "").strip()
    if not name:
        return "file"
    if "." in name:
        base, ext = name.rsplit(".", 1)
        return f"file_{_hash_token(base)}.{ext[:10]}"
    return f"file_{_hash_token(name)}"
