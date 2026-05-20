"""Register all rules at import. Call load_all() once at app startup."""

from src.consolidator.rules.authority.basic import (
    ExactAuthorityIdMatch,
    ExactNameAnyCountryAuthority,
    ExactNameCountryMatchAuthority,
    FuzzyNameSameCountryAuthority,
)
from src.consolidator.rules.authority.embedding_similarity import (
    EmbeddingCosineSameAuthority,
)
from src.consolidator.rules.authority.enrichment import TranslationEnrichmentAuthority
from src.consolidator.rules.contract.enrichment import TranslationEnrichmentContract
from src.consolidator.rules.company.exact_identifiers import (
    ExactCikMatch,
    ExactLeiMatch,
    ExactVatMatch,
)
from src.consolidator.rules.company.fuzzy import FuzzyNameSameCountry
from src.consolidator.rules.company.name_country import ExactNameCountryMatch
from src.consolidator.rules.company.successor import SuccessorLeiMatch
from src.consolidator.rules.gds.node_similarity import (
    GdsNodeSimilarityAuthority,
    GdsNodeSimilarityCompany,
)
from src.consolidator.rules.gds.wcc_collapse import (
    GdsSameAsClusterCollapseAuthority,
    GdsSameAsClusterCollapseCompany,
)
from src.consolidator.rules.registry import register


# Idempotency flag for the at-startup load — keeping it lowercase
# matches the mutable-cache convention used elsewhere in the package
# (eventlog._log_singleton, neo4j.client._driver).
_loaded = False  # pylint: disable=invalid-name


def load_all() -> None:
    global _loaded  # pylint: disable=global-statement
    if _loaded:
        return
    _loaded = True
    # Company — confidence-ordered via the registry's sort
    register(ExactLeiMatch())
    register(ExactCikMatch())
    register(ExactVatMatch())
    # Succession consolidation (active + retired LEI → merge, preserve retired
    # in historic_leis). 0.98 confidence — runs after exact-id rules.
    register(SuccessorLeiMatch())
    register(GdsSameAsClusterCollapseCompany())
    register(ExactNameCountryMatch())
    register(FuzzyNameSameCountry())
    register(GdsNodeSimilarityCompany())
    # Authority
    # Enrichment runs first (confidence 1.0, tie-broken by insertion order):
    # if the authority gets merged later in the same run, combined-property
    # semantics preserve translations on the canonical node.
    register(TranslationEnrichmentAuthority())
    register(ExactAuthorityIdMatch())
    register(GdsSameAsClusterCollapseAuthority())
    register(ExactNameCountryMatchAuthority())
    # Cross-country same-name flag for EU bodies (EEAS, JRC, eu-LISA, …).
    # Confidence 0.90 — runs after same-country exact (0.95) and below fuzzy.
    register(ExactNameAnyCountryAuthority())
    # LaBSE cosine-similarity flag (confidence 0.87). Catches cross-
    # lingual duplicates (e.g. Italian / English names of the same
    # ministry) that string rules miss. Never auto-merges.
    register(EmbeddingCosineSameAuthority())
    register(FuzzyNameSameCountryAuthority())
    register(GdsNodeSimilarityAuthority())
    # Contract — v1 multilingual: title-only translation.
    register(TranslationEnrichmentContract())
