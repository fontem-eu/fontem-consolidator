"""Pure-Python replica of Neo4j's apoc.text.clean.

The name_country resolver tier matches iff apoc.text.clean(a) ==
apoc.text.clean(b), so reproducing that function exactly lets the recall
eval run in-process. Validated to 100% (399/399) against materialised
Company.name_clean values pulled from prod across DEU/FRA/ITA/HUN/POL/
ESP/SWE/CZE/NLD/GBR.

Behaviour observed from prod:
  "Mészáros és Mészáros Zrt."  -> "meszarosesmeszaroszrt"   (accents fold)
  "Müller KG"                  -> "muellerkg"               (ü -> ue, German)
  "Société Générale S.A."      -> "societegeneralesa"
  "Łódź Sp. z o.o."            -> "łodzspzoo"               (ł kept; ó,ź fold)
"""
from __future__ import annotations

import unicodedata

# German umlaut transliteration must run BEFORE NFKD (which would otherwise
# strip ü to a bare u). apoc emits ü->ue / ö->oe / ä->ae / ß->ss.
_GERMAN = {
    "ä": "ae", "ö": "oe", "ü": "ue",
    "Ä": "ae", "Ö": "oe", "Ü": "ue", "ß": "ss",
}


def clean(value: str | None) -> str | None:
    """Lowercase, fold diacritics, strip every non-alphanumeric character.

    Returns None for None so callers can distinguish "absent" from "empty".
    """
    if value is None:
        return None
    for src, dst in _GERMAN.items():
        value = value.replace(src, dst)
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in without_marks.lower() if c.isalnum())
