"""Register all rules at import. Call load_all() once at app startup."""

from src.consolidator.rules.authority.basic import (
    ExactAuthorityIdMatch,
    ExactNameCountryMatchAuthority,
    FuzzyNameSameCountryAuthority,
)
from src.consolidator.rules.company.exact_identifiers import (
    ExactCikMatch,
    ExactLeiMatch,
    ExactVatMatch,
)
from src.consolidator.rules.company.fuzzy import FuzzyNameSameCountry
from src.consolidator.rules.company.name_country import ExactNameCountryMatch
from src.consolidator.rules.registry import register


_loaded = False


def load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    # Company — confidence-ordered via the registry's sort
    register(ExactLeiMatch())
    register(ExactCikMatch())
    register(ExactVatMatch())
    register(ExactNameCountryMatch())
    register(FuzzyNameSameCountry())
    # Authority
    register(ExactAuthorityIdMatch())
    register(ExactNameCountryMatchAuthority())
    register(FuzzyNameSameCountryAuthority())
