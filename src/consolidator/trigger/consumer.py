"""ConsolidatorTrigger — gmr-events EventConsumer that POSTs each
input event to the consolidator's ``/events/dispatch`` webhook.

Cycle prevention via ``INPUT_TYPES``: only Upsert*/Delete* events
are dispatched. The consolidator's own outputs (AssertSameAs,
RetractSameAs) and bracket markers (Begin/EndGraphReplace) advance
the offset without dispatch — the loop is broken by construction.

Concurrency: each batch is dispatched through a bounded thread
pool (``CONSOLIDATOR_TRIGGER_CONCURRENCY``, default 10). Order
doesn't matter inside a batch — each consolidate() runs against
its own entity in Neo4j and emits MERGE-based events, which are
idempotent. The whole batch's offset only commits once every
event in it has succeeded; if any one fails the batch retries
(individual successful events are no-ops on retry).

Failure handling:
  * 2xx, 409                → success.
  * 5xx, network error      → raise; EventConsumer retries the batch.
  * 4xx (other than 409)    → raise; eventually DLQ'd via batch retry.

Per-event DLQ would require extending the EventConsumer base class
to commit per-event rather than per-batch. We accept batch-level
DLQ — payload-shape errors are rare and surface fast.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
from fontem_event_schemas import EventEnvelope
from fontem_events.consumer import EventConsumer

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
    via a bounded HTTP fan-out per batch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.consolidator_url = os.environ["CONSOLIDATOR_URL"].rstrip("/")
        self.timeout = float(
            os.environ.get("CONSOLIDATOR_HTTP_TIMEOUT", "60")
        )
        self.concurrency = max(1, int(
            os.environ.get("CONSOLIDATOR_TRIGGER_CONCURRENCY", "10")
        ))

    def handle(self, batch: list[EventEnvelope]) -> None:
        # Filter early — we don't want to spend executor slots on
        # events the trigger will skip.
        to_dispatch = [ev for ev in batch if ev.event_type in INPUT_TYPES]
        skipped = len(batch) - len(to_dispatch)
        if not to_dispatch:
            if skipped:
                logger.debug(
                    "batch: skipped=%d (all non-INPUT_TYPES)", skipped,
                )
            return

        # One httpx.Client per worker so connection pools don't
        # serialise between threads. The Client itself is thread-safe
        # for concurrent requests, but a single shared pool of size
        # ~10 against the same target is fine — keep it simple.
        # Limits set just under the worker count to leave headroom.
        limits = httpx.Limits(
            max_connections=self.concurrency,
            max_keepalive_connections=self.concurrency,
        )
        with httpx.Client(timeout=self.timeout, limits=limits) as client:
            with ThreadPoolExecutor(
                max_workers=self.concurrency,
                thread_name_prefix="trigger",
            ) as pool:
                # list() forces all futures to surface their results
                # (and raise on first failure).
                results = list(pool.map(
                    lambda ev: self._dispatch(client, ev),
                    to_dispatch,
                ))
        logger.debug(
            "batch: dispatched=%d skipped=%d (concurrency=%d)",
            len(results), skipped, self.concurrency,
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
