"""Fuzzy name rule — flag (:SAME_AS) candidates within the same country when
the normalised names are Jaro-Winkler-similar above the threshold.

Pipeline:
  1. Use the Company.name fulltext index to fetch candidate :Company nodes
     in the same country (cheap retrieval — we still cap to 20).
  2. Normalise both names: upper-case, drop legal-form suffixes (AB, GmbH,
     SAS, LLP, LTD, ...), drop punctuation, collapse whitespace.
  3. Compute Jaro-Winkler similarity on the normalised pair.
  4. Skip if hard identifiers disagree (different valid LEI / CIK / VAT) —
     those pairs are definitely different entities, no point flagging.
  5. Emit :SAME_AS at the actual similarity score (not clamped) when
     sim >= settings.fuzzy_name_threshold.
"""

import re

from loguru import logger
from rapidfuzz.distance import JaroWinkler

from src.config import settings
from src.consolidator.rules.base import Candidate, Decision, Entity, Rule
from src.consolidator.rules.conflict import find_conflict

_LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')

# Legal-form tokens to strip before similarity (case-insensitive word-boundary).
# Keep this conservative — stripping too much causes different entities to look
# identical ("Acme AB" vs "Acme Ltd" should still compare as "ACME" = "ACME" → 1.0
# which is the correct call since hard-id conflict detection handles the
# "actually different legal entity" case).
# Alternation order matters — longer patterns FIRST so "S.A.R.L." wins over
# the shorter "S.A" prefix match.
#
# Cyrillic boilerplate is matched case-insensitively with an explicit flag
# block; Python's re only lowercases ASCII unless you opt in to Unicode.
_LEGAL_SUFFIX = re.compile(
    r"(?:"
    # Long forms first — they must win before short tokens like ТОВ / SIA / AS.
    # A missing entry here is NOT cosmetic: the unstripped boilerplate becomes
    # a shared prefix, and Jaro-WINKLER pays a bonus for shared prefixes, so
    # every company in that jurisdiction scores ~0.95 against every other. The
    # Latvian long form was absent and produced 29,204 edges, including
    # AR TEXTIL = ZABBIX at 0.953. `_shared_prefix_ok` is the structural
    # backstop for whatever is still missing from this list.
    r"(?i:ТОВАРИСТВО\s+З\s+ОБМЕЖЕНОЮ\s+ВІДПОВІДАЛЬНІСТЮ)|"       # UA LLC
    r"(?i:ОБЩЕСТВО\s+С\s+ОГРАНИЧЕННОЙ\s+ОТВЕТСТВЕННОСТЬЮ)|"      # RU LLC
    r"(?i:SPÓŁKA\s+Z\s+OGRANICZONĄ\s+ODPOWIEDZIALNOŚCIĄ)|"        # PL LLC long form
    r"(?i:SPÓŁKA\s+AKCYJNA)|"                                      # PL joint-stock
    r"(?i:SABIEDRĪBA\s+AR\s+IEROBEŽOTU\s+ATBILDĪBU)|"             # LV LLC
    r"(?i:AKCIJU\s+SABIEDRĪBA)|"                                   # LV joint-stock
    r"(?i:UŽDAROJI\s+AKCINĖ\s+BENDROVĖ)|"                         # LT LLC
    r"(?i:VIEŠOJI\s+ĮSTAIGA)|"                                     # LT public body
    r"(?i:AKCINĖ\s+BENDROVĖ)|"                                     # LT joint-stock
    r"(?i:SPOLEČNOST\s+S\s+RUČENÍM\s+OMEZENÝM)|"                  # CZ LLC
    r"(?i:AKCIOVÁ\s+SPOLEČNOST)|"                                  # CZ joint-stock
    r"(?i:SPOLOČNOSŤ\s+S\s+RUČENÍM\s+OBMEDZENÝM)|"                # SK LLC
    r"(?i:GESELLSCHAFT\s+MIT\s+BESCHRÄNKTER\s+HAFTUNG)|"          # DE LLC
    r"(?i:AKTIENGESELLSCHAFT)|"                                     # DE joint-stock
    r"(?i:SOCIETÀ\s+A\s+RESPONSABILITÀ\s+LIMITATA)|"              # IT LLC
    r"(?i:SOCIETÀ\s+PER\s+AZIONI)|"                               # IT joint-stock
    r"(?i:SOCIEDAD\s+(?:DE\s+)?RESPONSABILIDAD\s+LIMITADA)|"      # ES LLC
    r"(?i:SOCIEDAD\s+(?:LIMITADA|ANÓNIMA))|"                       # ES short-long
    r"(?i:SOCIEDADE\s+POR\s+QUOTAS)|"                             # PT LLC
    r"(?i:SOCIÉTÉ\s+(?:À|A)\s+RESPONSABILIT(?:É|E)\s+LIMIT(?:É|E)E)|"  # FR LLC
    r"(?i:SOCIÉTÉ\s+PAR\s+ACTIONS\s+SIMPLIFIÉE)|"                 # FR SAS long
    r"(?i:SOCIÉTÉ\s+ANONYME)|"                                     # FR SA long
    r"(?i:KORLÁTOLT\s+FELELŐSSÉGŰ\s+TÁRSASÁG)|"                    # HU LLC
    r"(?i:SOCIETATE\s+CU\s+R(?:Ă|A)SPUNDERE\s+LIMITAT(?:Ă|A))|"   # RO LLC
    r"(?i:БЪЛГАРСКО\s+ДРУЖЕСТВО\s+С\s+ОГРАНИЧЕНА\s+ОТГОВОРНОСТ)|"  # BG LLC
    r"(?i:ДРУЖЕСТВО\s+С\s+ОГРАНИЧЕНА\s+ОТГОВОРНОСТ)|"             # BG LLC (short)
    r"(?i:NAAMLOZE\s+VENNOOTSCHAP)|"                               # NL/BE NV
    r"(?i:BESLOTEN\s+VENNOOTSCHAP)|"                               # NL/BE BV
    r"(?i:OSAÜHING)|"                                               # EE LLC
    r"(?i:AKCIONARSKO\s+DRUŠTVO)|"                                 # RS/HR joint-stock
    r"(?i:DRUŠTVO\s+S\s+OGRANIČENOM\s+ODGOVORNOŠĆU)|"             # HR LLC
    # Short forms — word-boundary anchored
    r"\b(?:"
    # French / Romance
    r"SARL|S\.?A\.?R\.?L\.?|"
    r"S\.?A\.?S\.?|SAS|"
    r"SPRL|SRL|SPA|"
    # English/German/Nordic/Baltic
    r"GMBH|LTDA|LLP|LLC|LTD|"
    r"OYJ|ASA|APS|PLC|PTE|PTY|"
    r"S\.?A\.?|"
    r"AB|AG|AS|BV|INC|KFT|KG|KK|NV|OY|SE|SL|SLU|SP|UG|"
    # Baltic
    r"UAB|SIA|OÜ|"
    # Czech / Slovak
    r"S\.?R\.?O\.?|SRO|A\.?S\.?|"
    # Polish short
    r"SP\.?\s?Z\s?O\.?\s?O\.?|S\.?C\.?|"
    # Cyrillic short (Ukrainian ТОВ, Russian ООО)
    r")\.?\b|"
    r"(?i:\bТОВ\b)|(?i:\bООО\b)"
    r")",
    re.UNICODE,
)
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def _lucene_sanitize(name: str) -> str:
    return _LUCENE_SPECIAL.sub(" ", name).strip()


def _normalise(name: str) -> str:
    s = name.upper()
    s = _LEGAL_SUFFIX.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return s


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _shared_prefix_ok(a: str, b: str, threshold: float) -> bool:
    """Reject pairs whose similarity rests on shared leading boilerplate.

    `_LEGAL_SUFFIX` is a wordlist, and a wordlist is never complete. When
    it misses a jurisdiction's long legal form, that form survives
    normalisation as a shared prefix on every company in the country — and
    because Jaro-*Winkler* adds a bonus for shared prefixes, the score goes
    UP the longer the boilerplate is. That is how one Latvian company was
    asserted identical to 1,333 others at ~0.95.

    So: when the two names share a long prefix, treat the prefix as
    boilerplate and require what FOLLOWS it to clear the bar on its own.

    A remainder that is empty on either side is the legitimate
    "name" vs "name + qualifier" case ("PI VINDIJA DD" vs
    "PI VINDIJA DD VARAZDIN") and is left to the main score.
    """
    lcp = _common_prefix_len(a, b)
    if lcp < settings.fuzzy_shared_prefix_guard_chars:
        return True
    rest_a, rest_b = a[lcp:].strip(), b[lcp:].strip()
    if not rest_a or not rest_b:
        return True
    return JaroWinkler.normalized_similarity(rest_a, rest_b) >= threshold


class FuzzyNameSameCountry(Rule):
    name = "fuzzy_name_same_country"
    description = (
        "Full-text candidate fetch + Jaro-Winkler similarity on normalised "
        "names (same country). Flags :SAME_AS above "
        f"{settings.fuzzy_name_threshold}. Pairs with disagreeing hard IDs are skipped."
    )
    entity_types = {"Company"}
    confidence = 0.95  # upper bound; emitted confidence is the actual Jaro-Winkler score
    action = "flag"
    # Canary on 1k Companies showed the band ≥0.97 to be exclusively
    # legal-form-stripped variants ("BAYARD" ↔ "Bayard SAS",
    # "SARL PROJARDIN" ↔ "PROJARDIN") — same entity, different
    # registration form. Below 0.97 the queue starts mixing in
    # parent/subsidiary cases ("Mercedes-Benz Leasing" ↔ "Mercedes-Benz
    # AG") which are NOT mergeable. Auto-merge top tier; flag the rest.
    auto_merge_threshold = 0.97

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("name")) and bool(entity.properties.get("country"))

    # The retrieval / normalisation / scoring pipeline keeps every
    # intermediate (sanitized, threshold, normalised pair, similarity)
    # readable on one screen; splitting it would just push the locals
    # behind helper boundaries.
    async def find_candidates(self, entity: Entity) -> list[Candidate]:  # pylint: disable=too-many-locals
        # Imported lazily so unit tests patching the module-level
        # `get_driver` see the patched callable rather than a name
        # already bound at import time.
        from src.consolidator.neo4j.client import get_driver  # pylint: disable=import-outside-toplevel
        driver = await get_driver()
        sanitized = _lucene_sanitize(entity.properties["name"])
        if not sanitized:
            return []
        async with driver.session() as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes('company_name_ft', $search_text)
                    YIELD node, score
                    WHERE node.gmr_id <> $self_id
                      AND node.country = $country
                    RETURN node, score
                    LIMIT 20
                    """,
                    search_text=sanitized,
                    self_id=entity.id,
                    country=entity.properties["country"],
                )
                records = [record async for record in result]
            # Ephemeral test DBs may not have the company_name_ft fulltext
            # index — log + treat as "no candidates" rather than aborting.
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("fuzzy_name: fulltext query failed: {}", exc)
                return []

        threshold = settings.fuzzy_name_threshold
        norm_self = _normalise(entity.properties["name"])
        # Too short to discriminate — see settings.fuzzy_min_distinctive_chars.
        if len(norm_self) < settings.fuzzy_min_distinctive_chars:
            return []

        out: list[Candidate] = []
        for rec in records:
            props = dict(rec["node"])
            norm_other = _normalise(props.get("name") or "")
            if len(norm_other) < settings.fuzzy_min_distinctive_chars:
                continue
            sim = JaroWinkler.normalized_similarity(norm_self, norm_other)
            if sim < threshold:
                continue
            if not _shared_prefix_ok(norm_self, norm_other, threshold):
                continue
            out.append(
                Candidate(
                    entity=Entity("Company", props["gmr_id"], props),
                    context={"jw_similarity": sim, "raw_score": float(rec["score"])},
                )
            )
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        # If the two nodes disagree on a canonicalisable hard identifier,
        # they are definitely different entities. Skip without writing.
        if find_conflict(entity, candidate) is not None:
            return Decision(
                rule_name=self.name,
                action="noop",
                source_id=entity.id,
                target_id=candidate.entity.id,
                confidence=0.0,
                entity_type="Company",
                details={
                    "skipped": "hard_id_conflict",
                    "jw_similarity": candidate.context.get("jw_similarity"),
                },
            )
        sim = float(candidate.context.get("jw_similarity", 0.0))
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=sim,
            entity_type="Company",
            details={"jw_similarity": sim},
        )
