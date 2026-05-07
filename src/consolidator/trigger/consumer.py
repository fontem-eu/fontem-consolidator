"""ConsolidatorTrigger — gmr-events EventConsumer that POSTs each
input event to the consolidator's ``/events/dispatch`` webhook.

Cycle prevention via ``INPUT_TYPES``: only Upsert*/Delete* events
are dispatched. The consolidator's own outputs (AssertSameAs,
RetractSameAs) and bracket markers (Begin/EndGraphReplace) advance
the offset without dispatch — the loop is broken by construction.

Failure handling:
  * 2xx, 409                → success, advance offset.
  * 5xx, network error      → raise; EventConsumer retries the batch.
  * 4xx (other than 409)    → raise; eventually DLQ'd via batch retry.

Per-event DLQ would require extending the EventConsumer base class
to commit per-event rather than per-batch. For Phase D we accept
batch-level DLQ — payload-shape errors are rare and surface fast.
"""
from __future__ import annotations

import logging
import os

import httpx
from gmr_event_schemas import EventEnvelope
from gmr_events.consumer import EventConsumer

logger = logging.getLogger(__name__)

# The consolidator only acts on entity events. Outputs from the
# consolidator (AssertSameAs / RetractSameAs) are intentionally
# excluded so the trigger doesn't re-dispatch its own writes.
# Control events (Begin/EndGraphReplace) carry no entity to
# consolidate.
INPUT_TYPES: frozenset[str] = frozenset({
    "UpsertCompany",
    "UpsertContract",
    "UpsertAuthority",
    "UpsertFiling",
    "DeleteCompany",
    "DeleteContract",
    "DeleteAuthority",
    "DeleteFiling",
})


class ConsolidatorTrigger(EventConsumer):
    """Walks ``events.entity_events`` and triggers consolidation
    one entity at a time."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.consolidator_url = os.environ["CONSOLIDATOR_URL"].rstrip("/")
        self.timeout = float(
            os.environ.get("CONSOLIDATOR_HTTP_TIMEOUT", "60")
        )

    def handle(self, batch: list[EventEnvelope]) -> None:
        skipped = 0
        dispatched = 0
        with httpx.Client(timeout=self.timeout) as client:
            for ev in batch:
                if ev.event_type not in INPUT_TYPES:
                    skipped += 1
                    continue
                self._dispatch(client, ev)
                dispatched += 1
        if skipped:
            logger.debug(
                "batch: dispatched=%d skipped=%d (non-INPUT_TYPES)",
                dispatched, skipped,
            )

    def _dispatch(
        self, client: httpx.Client, ev: EventEnvelope,
    ) -> None:
        body = {
            "seq": ev.seq,
            "event_type": ev.event_type,
            "iri": ev.iri,
            "domain": ev.domain,
            "payload": ev.payload,
            "batch_id": str(ev.batch_id) if ev.batch_id else None,
            "producer": ev.producer,
        }
        url = f"{self.consolidator_url}/events/dispatch"
        r = client.post(url, json=body)
        # 409 means "already done" — idempotent success.
        if r.status_code == 409:
            logger.info(
                "dispatch seq=%s type=%s → 409 (treated as success)",
                ev.seq, ev.event_type,
            )
            return
        r.raise_for_status()
        logger.info(
            "dispatch seq=%s type=%s → %s",
            ev.seq, ev.event_type, r.status_code,
        )
