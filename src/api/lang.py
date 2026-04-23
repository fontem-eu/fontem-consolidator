"""Language helpers for API handlers.

The :Authority enrichment rule populates `name_<lang>` for 24 EU ISO-639-1
codes. This module whitelists caller-provided `lang` query params and
provides a helper to apply the translation when projecting entity props
to the response body.
"""
from __future__ import annotations

from typing import Final

# Must stay in sync with:
#   - src/consolidator/clients/linguistics.py  EU_OFFICIAL_LANGS
#   - gmr-linguistics  /languages
#   - gmr-web          src/composables/eu-languages.js
EU_LANGS: Final[frozenset[str]] = frozenset({
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt",
    "ro", "sk", "sl", "sv",
})


def safe_lang(value: str | None) -> str | None:
    """Return a whitelisted EU language code or None.

    Accepts "EN", "pt-BR", "  de ", etc. Strips to the first two
    lowercase letters before checking. Anything else → None.
    """
    if not value or not isinstance(value, str):
        return None
    code = value.strip()[:2].lower()
    return code if code in EU_LANGS else None


def apply_translation(props: dict, lang: str | None) -> dict:
    """Return a shallow copy of `props` with `name` swapped for
    `name_<lang>` when the translation is present. Leaves `props`
    untouched if `lang` is None or the translation is missing.

    Safe for any entity — Company nodes don't carry `name_<lang>`
    properties, so the swap is a no-op.
    """
    if not lang:
        return props
    translated = props.get(f"name_{lang}")
    if not translated:
        return props
    out = dict(props)
    out["name"] = translated
    return out
