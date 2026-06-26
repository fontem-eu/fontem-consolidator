from src.consolidator.rules.base import Candidate, Decision, Entity, Rule
from src.consolidator.rules.conflict import conflict_decision, find_conflict


class ExactNameCountryMatch(Rule):
    name = "exact_name_country_match"
    description = (
        "Two :Company nodes share the same (name, country) after whitespace/"
        "punctuation normalisation (apoc.text.clean) → FLAG for human review. "
        "Name+country alone has a measured false-merge floor of ~0.27% (even at "
        "40+ chars: genuinely different entities sharing generic institutional "
        "names — 'Municipal District Heating Company', 'Diocesan Institute for "
        "the Support of the Clergy'), and ~68% of same-name company pairs are "
        "distinct registered entities. That is far above the 0.01% bar, and no "
        "corroborating signal is available on the name-only side (0% postal_code "
        "coverage), so this rule NEVER auto-merges — only the hard-identifier "
        "rules (exact LEI/VAT/CIK) do. Revisit if a distinctiveness or graph-"
        "structure corroboration signal can close the gap."
    )
    entity_types = {"Company"}
    confidence = 0.95
    # Flag-only: even with settings.auto_merge_enabled on, this stays in the
    # review queue. auto_merge_threshold is left None so the engine's flag->merge
    # promotion never fires for it.
    action = "flag"

    async def applies(self, entity: Entity) -> bool:
        return bool(entity.properties.get("name")) and bool(entity.properties.get("country"))

    async def find_candidates(self, entity: Entity) -> list[Candidate]:
        # Imported lazily so unit tests patching the module-level
        # `get_driver` see the patched callable rather than a name
        # already bound at import time.
        from src.consolidator.neo4j.client import get_driver  # pylint: disable=import-outside-toplevel
        driver = await get_driver()
        async with driver.session() as session:
            # apoc.text.clean strips non-alphanumerics + lowercases, so
            # "NEURAXPHARM FRANCE ( Rang 1)" == "NEURAXPHARM FRANCE (Rang 1)"
            # == "NEURAXPHARM FRANCE  (Rang 1)" all collapse. We match
            # against the sink-materialised ``name_clean`` property so
            # the index on Company.name_clean keeps this O(log N).
            result = await session.run(
                """
                MATCH (c:Company)
                WHERE c.name_clean = apoc.text.clean($name)
                  AND c.country = $country
                  AND c.gmr_id <> $self_id
                RETURN c
                """,
                name=entity.properties["name"],
                country=entity.properties["country"],
                self_id=entity.id,
            )
            records = [record async for record in result]
        out = []
        for rec in records:
            props = dict(rec["c"])
            out.append(Candidate(entity=Entity("Company", props["gmr_id"], props), context={}))
        return out

    async def resolve(self, entity: Entity, candidate: Candidate) -> Decision:
        # A hard-id disagreement is a conflict-flag (mismatched LEI/VAT); a clean
        # name match is a plain review flag. Neither auto-merges — name+country
        # is too ambiguous to fuse identities automatically (see class docstring).
        conflict = find_conflict(entity, candidate)
        if conflict:
            return conflict_decision(
                rule_name=self.name,
                entity=entity,
                candidate=candidate,
                confidence=self.confidence,
                conflict=conflict,
            )
        return Decision(
            rule_name=self.name,
            action="flag",
            source_id=entity.id,
            target_id=candidate.entity.id,
            confidence=self.confidence,
            entity_type="Company",
            details={"name_country_review": True},
        )
