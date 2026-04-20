from src.consolidator.rules.base import Rule

_REGISTRY: list[Rule] = []


def register(rule: Rule) -> Rule:
    _REGISTRY.append(rule)
    return rule


def list_rules() -> list[Rule]:
    return sorted(_REGISTRY, key=lambda r: -r.confidence)
