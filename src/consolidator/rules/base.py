from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Action = Literal["merge", "link", "flag", "noop", "enrich"]


@dataclass
class Entity:
    entity_type: str  # "Company" | "Authority"
    id: str           # gmr_id or authority_id
    properties: dict


@dataclass
class Candidate:
    entity: Entity
    context: dict  # rule-specific hints (e.g. {"matched_lei": "..."})


@dataclass
class Decision:
    rule_name: str
    action: Action
    source_id: str
    target_id: str
    confidence: float
    entity_type: str
    details: dict


class Rule(ABC):
    name: str
    entity_types: set[str]
    confidence: float
    action: Action
    description: str = ""
    # When set, the engine upgrades a `flag` decision to `merge` if
    # `decision.confidence >= auto_merge_threshold` AND the decision
    # carries no conflict signal AND `settings.auto_merge_enabled` is
    # on. None = default flag-only behaviour. Calibrated from the
    # canary sweeps — see commit history on the rule classes for the
    # data those numbers came from.
    auto_merge_threshold: float | None = None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "entity_types": sorted(self.entity_types),
            "confidence": self.confidence,
            "action": self.action,
            "auto_merge_threshold": self.auto_merge_threshold,
            "description": self.description,
        }

    @abstractmethod
    async def applies(self, entity: Entity) -> bool:
        """Cheap predicate — does this rule have a chance of firing for this entity."""

    @abstractmethod
    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        """Cypher/GDS lookup for potential matches."""

    @abstractmethod
    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        """Produce a Decision for an (entity, candidate) pair."""
