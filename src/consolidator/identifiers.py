"""Canonical forms for hard identifiers.

Each `canon_*` function returns a canonical string if the raw value matches
the expected format for that identifier type, or `None` otherwise. Conflict
detection compares canonicals, so malformed values (TED notice IDs stuffed
into the `vat` field, whitespace-padded IDs, wrong-length numbers) are
treated as "unknown" rather than disagreement.

The originals are never rewritten on the node — this module only affects
comparison.
"""

from __future__ import annotations

import re

# Country-specific VAT patterns. Lowercase keys, uppercase regex content.
# Patterns are intentionally strict — when in doubt, return None so the
# conflict rule falls back to "no disagreement" rather than a false positive.
_VAT_BY_COUNTRY = {
    "AT": re.compile(r"^ATU\d{8}$"),
    "BE": re.compile(r"^BE0?\d{9,10}$"),
    "BG": re.compile(r"^BG\d{9,10}$"),
    "CY": re.compile(r"^CY\d{8}[A-Z]$"),
    "CZ": re.compile(r"^CZ\d{8,10}$"),
    "DE": re.compile(r"^DE\d{9}$"),
    "DK": re.compile(r"^DK\d{8}$"),
    "EE": re.compile(r"^EE\d{9}$"),
    "EL": re.compile(r"^EL\d{9}$"),      # Greece (VIES prefix)
    "GR": re.compile(r"^GR\d{9}$"),
    "ES": re.compile(r"^ES[A-Z0-9]\d{7}[A-Z0-9]$"),
    "FI": re.compile(r"^FI\d{8}$"),
    "FR": re.compile(r"^FR[A-Z0-9]{2}\d{9}$"),
    "GB": re.compile(r"^GB(\d{9}|\d{12}|GD\d{3}|HA\d{3})$"),
    "HR": re.compile(r"^HR\d{11}$"),
    "HU": re.compile(r"^HU\d{8}$"),
    "IE": re.compile(r"^IE\d[A-Z0-9\+\*]\d{5}[A-Z]{1,2}$"),
    "IT": re.compile(r"^IT\d{11}$"),
    "LT": re.compile(r"^LT(\d{9}|\d{12})$"),
    "LU": re.compile(r"^LU\d{8}$"),
    "LV": re.compile(r"^LV\d{11}$"),
    "MT": re.compile(r"^MT\d{8}$"),
    "NL": re.compile(r"^NL\d{9}B\d{2}$"),
    "PL": re.compile(r"^PL\d{10}$"),
    "PT": re.compile(r"^PT\d{9}$"),
    "RO": re.compile(r"^RO\d{2,10}$"),
    "SE": re.compile(r"^SE\d{12}$"),
    "SI": re.compile(r"^SI\d{8}$"),
    "SK": re.compile(r"^SK\d{10}$"),
}

# French SIRET — 14 digits. Commonly written into `vat` for French entities.
_SIRET_RE = re.compile(r"^\d{14}$")

# TED notice-like identifiers the ETL sometimes writes into `vat` by mistake.
_TED_NOTICE_RE = re.compile(r"^\d+-\d+-\d+-\d+$")

_LEI_RE = re.compile(r"^[A-Z0-9]{20}$")


def _strip_punct(raw: str) -> str:
    """Remove whitespace, dots, hyphens. Keep alphanumerics only."""
    return re.sub(r"[\s.\-/]", "", raw).upper()


def canon_lei(raw: str | None) -> str | None:
    """LEI is a 20-char alphanumeric. Reject anything that doesn't match."""
    if not raw:
        return None
    s = raw.strip().upper()
    return s if _LEI_RE.match(s) else None


def canon_cik(raw: str | None) -> str | None:
    """CIK is a number. Strip leading zeros so '0000123' and '123' match."""
    if not raw:
        return None
    s = raw.strip().lstrip("0") or "0"
    return s if s.isdigit() else None


def canon_vat(raw: str | None) -> str | None:
    """Canonical VAT: strip punctuation, match against EU country patterns.

    A value that looks like a TED notice or a standalone 14-digit SIRET is
    NOT a VAT — return None so the conflict check doesn't fire on it.
    """
    if not raw:
        return None
    s = _strip_punct(raw)
    if not s:
        return None
    for rx in _VAT_BY_COUNTRY.values():
        if rx.match(s):
            return s
    return None


def canon_siret(raw: str | None) -> str | None:
    """SIRET is a 14-digit number (French establishment)."""
    if not raw:
        return None
    s = _strip_punct(raw)
    return s if _SIRET_RE.match(s) else None


def looks_like_ted_notice(raw: str | None) -> bool:
    """True when the value has the TED publication-notice shape (`1234-5-6-1`)."""
    return bool(raw and _TED_NOTICE_RE.match(raw.strip()))
