"""A consolidator outage must never be mistaken for a poison event.

EventConsumer skips an event that fails ``max_attempts`` times in a
row. Nothing re-emits a skipped event, so a consolidator redeploy
would leave those entities un-consolidated for good — no SAME_AS
candidate ever proposed for them. virtuoso_sink lost 4,006 events to
exactly this on 2026-09-06.
"""
# pylint: disable=protected-access,import-outside-toplevel
from unittest.mock import patch

import httpx
import pytest


@pytest.fixture(name="trigger")
def _trigger(monkeypatch):
    monkeypatch.setenv("CONSOLIDATOR_URL", "http://consolidator.test")
    from src.consolidator.trigger.consumer import ConsolidatorTrigger

    with patch("src.consolidator.trigger.consumer.EventConsumer.__init__",
               lambda self, *a, **k: None):
        t = ConsolidatorTrigger.__new__(ConsolidatorTrigger)
        t.consolidator_url = "http://consolidator.test"
        t.timeout = 60.0
        t.concurrency = 10
    return t


def _status_error(code):
    request = httpx.Request("POST", "http://consolidator.test/events/dispatch")
    return httpx.HTTPStatusError(
        str(code), request=request,
        response=httpx.Response(code, request=request))


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("connection refused"),
    httpx.ReadTimeout("timed out"),
    httpx.RemoteProtocolError("server disconnected"),
])
def test_transport_failures_are_retryable(trigger, exc):
    """A redeploy is not a reason to drop the entity forever."""
    assert trigger.is_retryable(exc) is True


@pytest.mark.parametrize("code", [500, 502, 503, 504, 429])
def test_server_health_statuses_are_retryable(trigger, code):
    """5xx is the consolidator unhealthy; 429 is it asking us to slow
    down. Neither says anything about the event."""
    assert trigger.is_retryable(_status_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_stay_poison(trigger, code):
    """A 4xx other than 429 is the consolidator rejecting THIS event —
    the case the poison-skip exists for."""
    assert trigger.is_retryable(_status_error(code)) is False


def test_unrelated_exceptions_stay_poison(trigger):
    assert trigger.is_retryable(KeyError("gmr_id")) is False
    assert trigger.is_retryable(ValueError("bad payload")) is False
