"""Stop the trigger cleanly when Kubernetes asks.

The Deployment runs exec-form ``python -m src.consolidator.trigger``, so
the interpreter is PID 1 — and Linux does not apply default signal
actions to PID 1. A signal with no handler *registered* is ignored
outright, not fatal. Without the registration below SIGTERM does nothing
and the pod only dies when the grace period expires and the kernel sends
SIGKILL.

The sweeper beside this one already stops cleanly (``sweeper.run``
installs an asyncio stop event and drains its tasks). The trigger never
got the same treatment, and it is the synchronous half: its loop lives
in the vendored ``fontem_events`` wheel and takes no stop flag, so the
stop is a SystemExit rather than a cooperative flag. ``run_forever``
guards its body with ``except Exception`` and SystemExit derives from
BaseException, so it unwinds instead of being swallowed.

Why it matters beyond tidiness: kured drains a node with
``--drain-grace-period=1800`` and abandons the drain after
``--drain-timeout=35m`` with ``--force-reboot=false``. One container
ignoring SIGTERM burns 30 of those 35 minutes; a drain that times out
leaves the node cordoned, and prod's Postgres is on a node-local PV.

Both landing sites are safe. Between batches — the common case, since
the trigger idles in ``time.sleep(poll_interval)`` — the last offset is
already committed and nothing repeats. Mid-batch, the offset is
committed only after a batch succeeds, so those events are re-read on
the next start; consolidation is re-runnable by design, which is the
same replay the loop already performs when an iteration raises.
"""
import logging
import signal

logger = logging.getLogger(__name__)

STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def install_stop_handlers(register=signal.signal) -> None:
    """Register the stop handlers. ``register`` is injectable for tests."""

    def _stop(signum, _frame):
        logger.info("consolidator_trigger: signal %s received, stopping", signum)
        raise SystemExit(0)

    for sig in STOP_SIGNALS:
        register(sig, _stop)
