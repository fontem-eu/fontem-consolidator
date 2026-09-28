"""What the live title rules translate, and from which language.

Shared by the Contract and CohesionProject rules. Since 2026-09-28 they run
on Nebius and only above the owner's euro cut-offs (contracts EUR 250M,
grants EUR 70M of EU contribution); everything below is left to budgeted
backfill runs (``backfill_translations``).

The source is what the notice states (``title_lang``), else a language the
backfill runner detected and kept (``title_lang_detected``). A title with
neither is not translated live: the runner detects before it translates,
and a live rule must not guess — the country guess labelled English titles
Lithuanian, and the identify-it prompt once copied titles into every
language.
"""
from __future__ import annotations

from src.consolidator.clients.linguistics import EU_OFFICIAL_LANGS

#: BCP-47 "undetermined": a stated language outside the 24, which
#: linguistics has no name for, so the model identifies it.
UNDETERMINED = "und"


def source_language(props: dict) -> str | None:
    """The title's language, or None when nothing states or detected it.

    A detection of "und" means the title has no words to translate (codes,
    reference numbers), so it is None here too.
    """
    stated = (props.get("title_lang") or "").lower()
    if stated:
        return stated if stated in EU_OFFICIAL_LANGS else UNDETERMINED
    detected = (props.get("title_lang_detected") or "").lower()
    if detected and detected != UNDETERMINED:
        return detected if detected in EU_OFFICIAL_LANGS else UNDETERMINED
    return None


def title_changed(props: dict) -> bool:
    """The title differs from the one its translations were made from."""
    source = props.get("title_translated_from")
    return source is not None and source != props.get("title")


def missing_targets(props: dict) -> list[str]:
    """The EU languages to translate into: those not yet on the node, or all
    but the source when the title has changed since it was translated."""
    src = source_language(props)
    if src is None:
        return []
    retitled = title_changed(props)
    return [code for code in EU_OFFICIAL_LANGS
            if code != src and (retitled or not props.get(f"title_{code}"))]


def at_least(props: dict, prop: str, minimum: float) -> bool:
    """The node's value property is known and at or above the cut-off."""
    value = props.get(prop)
    return isinstance(value, (int, float)) and value >= minimum
