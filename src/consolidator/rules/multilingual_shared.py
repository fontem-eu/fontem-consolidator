"""Shared helpers for multilingual enrichment rules (Authority, Contract).

Both rules infer a source language when the node doesn't carry an explicit
one, then ask gmr-linguistics to translate into the remaining EU locales.
"""
from __future__ import annotations

from typing import Final

# Country (ISO-3166 alpha-3) → primary official language (ISO-639-1). Covers
# the 27 EU member states; non-EU buyers (UK/CH/US embassies appearing in
# TED) fall through to "en".
COUNTRY_PRIMARY_LANG: Final[dict[str, str]] = {
    "AUT": "de", "BEL": "nl", "BGR": "bg", "HRV": "hr", "CYP": "el",
    "CZE": "cs", "DNK": "da", "EST": "et", "FIN": "fi", "FRA": "fr",
    "DEU": "de", "GRC": "el", "HUN": "hu", "IRL": "en", "ITA": "it",
    "LVA": "lv", "LTU": "lt", "LUX": "fr", "MLT": "mt", "NLD": "nl",
    "POL": "pl", "PRT": "pt", "ROU": "ro", "SVK": "sk", "SVN": "sl",
    "ESP": "es", "SWE": "sv",
}


def source_lang_from_country(country: str | None) -> str:
    """Primary official language for a country code, or "en" as last resort."""
    if not country:
        return "en"
    return COUNTRY_PRIMARY_LANG.get(country.upper(), "en")
