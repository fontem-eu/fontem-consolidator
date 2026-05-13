"""Entrypoint for the consolidator-trigger Deployment.

Configured entirely via env (see ``fontem_events.consumer.ConsumerConfig``):

  EVENT_CONSUMER_NAME       — e.g. "consolidator_trigger"
  EVENTS_DATABASE_URL       — Postgres DSN for the event log
  EVENT_UPSTREAM_CONSUMER   — name of consumer to high-watermark
                              gate against (e.g. "neo4j_sink")
  EVENT_BATCH_SIZE          — default 1000
  EVENT_POLL_INTERVAL       — default 5.0s
  CONSOLIDATOR_URL          — http URL to the fontem-consolidator service
  CONSOLIDATOR_HTTP_TIMEOUT — default 60s
  METRICS_PORT              — default 9100
  KUMA_PUSH_URL             — optional heartbeat URL

The trigger is single-replica: ordering matters and the offset
table is per-consumer-name.
"""
import logging

from .consumer import ConsolidatorTrigger


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    trigger = ConsolidatorTrigger.from_env()
    trigger.run_forever()


if __name__ == "__main__":
    main()
