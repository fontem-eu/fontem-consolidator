"""Realistic name-variation generator — the recall ground truth.

Nothing is stored: variants are generated from a seed name on demand. Each
Perturbation is tagged ``clean_invariant`` — whether apoc.text.clean should
absorb it (accents/case/punctuation/whitespace) or not (word-level changes
like legal-form expansion or & -> "and"). That split is the point: the
clean-invariant set is a regression guard on clean(); the rest measures the
matcher's real blind spots (where only the fuzzy tier can recover the link),
calibrated to mirror the same legal-form/spelling drift seen in the real
VAT-variant pairs in the graph.
"""
from __future__ import annotations

from dataclasses import dataclass

# Abbreviated <-> expanded legal forms across the member states that dominate
# the graph. Expanding/abbreviating is the single most common reason a TED
# winner ("... Zrt.") fails to match its GLEIF twin ("... Zártkörűen Működő
# Részvénytársaság") — both clean to different strings.
LEGAL_FORMS = [
    ("Zrt.", "Zártkörűen Működő Részvénytársaság"),
    ("Kft.", "Korlátolt Felelősségű Társaság"),
    ("Nyrt.", "Nyilvánosan Működő Részvénytársaság"),
    ("S.r.l.", "Società a responsabilità limitata"),
    ("S.p.A.", "Società per Azioni"),
    ("GmbH", "Gesellschaft mit beschränkter Haftung"),
    ("AG", "Aktiengesellschaft"),
    ("SARL", "Société à responsabilité limitée"),
    ("S.A.", "Société Anonyme"),
    ("S.L.", "Sociedad Limitada"),
    ("Sp. z o.o.", "Spółka z ograniczoną odpowiedzialnością"),
    ("s.r.o.", "společnost s ručením omezeným"),
    ("a.s.", "akciová společnost"),
    ("B.V.", "Besloten Vennootschap"),
    ("Ltd", "Limited"),
]

_FOLD = {
    "á": "a", "à": "a", "â": "a", "ä": "a", "é": "e", "è": "e", "ê": "e",
    "í": "i", "ï": "i", "ó": "o", "ö": "o", "ő": "o", "ú": "u", "ü": "u",
    "ű": "u", "ç": "c", "ñ": "n", "ž": "z", "š": "s", "č": "c", "ř": "r",
    "ą": "a", "ę": "e", "ł": "l", "ń": "n", "ś": "s", "ź": "z", "ż": "z",
}


def _strip_accents(name: str) -> str:
    return "".join(_FOLD.get(c, _FOLD.get(c.lower(), c)) for c in name)


@dataclass(frozen=True)
class Perturbation:
    name: str           # the perturbed string
    kind: str           # what was changed
    clean_invariant: bool  # True iff apoc.text.clean should absorb it


def perturbations(name: str):
    """Yield realistic variants of ``name``. Deterministic (no RNG) so the
    eval is reproducible without seeding."""
    # --- clean-invariant noise (clean() must absorb these) ---
    yield Perturbation(name.upper(), "uppercase", True)
    yield Perturbation(name.lower(), "lowercase", True)
    # strip_accents is NOT clean-invariant: apoc folds ü->ue, so a human
    # dropping the umlaut to bare "u" ("Muller") no longer matches "Müller".
    yield Perturbation(_strip_accents(name), "strip_accents", False)
    yield Perturbation(name.replace(" ", "  "), "double_space", True)
    yield Perturbation(name.replace(".", "").replace(",", ""), "drop_punct", True)
    yield Perturbation(f"  {name} ", "edge_space", True)
    # --- word-level drift (clean() does NOT absorb; needs fuzzy/identifier) ---
    for abbr, full in LEGAL_FORMS:
        if abbr in name:
            yield Perturbation(name.replace(abbr, full), "expand_legal_form", False)
            break
        if full in name:
            yield Perturbation(name.replace(full, abbr), "abbrev_legal_form", False)
            break
    if " & " in name:
        yield Perturbation(name.replace(" & ", " and "), "ampersand_to_and", False)
