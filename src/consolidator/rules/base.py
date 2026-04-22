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

    def describe(self) -> dict:
        return {
            "name": self.name,
            "entity_types": sorted(self.entity_types),
            "confidence": self.confidence,
            "action": self.action,
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
