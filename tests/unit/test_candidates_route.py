"""Route-level unit tests for /candidates and /decisions."""

from unittest.mock import AsyncMock, MagicMock, patch


def _stub_session(run_results):
    session = AsyncMock()

    class _Result:
        def __init__(self, recs):
            self._recs = recs
            self._it = iter(recs)

        def __aiter__(self):
            self._it = iter(self._recs)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def single(self):
            return self._recs[0] if self._recs else None

    async def run(*args, **kwargs):  # pylint: disable=unused-argument
        return _Result(run_results.pop(0))

    session.run = AsyncMock(side_effect=run)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)

    driver = AsyncMock()
    driver.session = MagicMock(return_value=ctx)
    return driver


def test_list_candidates_empty(client):
    c, _ = client
    driver = _stub_session([[]])
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)):
        r = c.get("/candidates?reviewed=false&limit=10")
    assert r.status_code == 200
    assert r.json() == []


def test_list_candidates_returns_shape(client):
    c, _ = client
    driver = _stub_session(
        [
            [
                {
                    "confidence": 0.94,
                    "rule_name": "fuzzy_name_same_country",
                    "detected_at": "2026-04-20T12:00:00Z",
                    # Parallel arrays — Neo4j can't store list<map> on a
                    # relationship, so the route zips these on the way out.
                    "det_rules": [
                        "fuzzy_name_same_country",
                        "embedding_cosine_authority",
                    ],
                    "det_confs": [0.94, 0.87],
                    "det_dates": ["2026-04-20T12:00:00Z", "2026-04-20T12:05:00Z"],
                    "conflict": False,
                    # Why a pair is contested, not just that it is. The
                    # route surfaces these so a reviewer can see which
                    # identifier disagreed.
                    "conflict_property": None,
                    "conflict_left": None,
                    "conflict_right": None,
                    "a_labels": ["Company"],
                    "a_props": {"gmr_id": "gmr-A", "name": "Acme"},
                    "b_labels": ["Company"],
                    "b_props": {"gmr_id": "gmr-B", "name": "Acme Inc"},
                }
            ]
        ]
    )
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)):
        r = c.get("/candidates?reviewed=false")
    body = r.json()
    assert len(body) == 1
    cand = body[0]
    # Legacy summary fields stay populated for backward compat.
    assert cand["rule_name"] == "fuzzy_name_same_country"
    assert cand["confidence"] == 0.94
    assert cand["from_id"] == "gmr-A"
    assert cand["to_id"] == "gmr-B"
    assert cand["entity_type"] == "Company"
    # Multi-rule detections list is the new field reviewers consume.
    assert len(cand["detections"]) == 2
    rules = {d["rule_name"] for d in cand["detections"]}
    assert rules == {"fuzzy_name_same_country", "embedding_cosine_authority"}


def test_list_candidates_back_fills_detections_for_legacy_edges(client):
    """Edges written before the multi-rule schema have no
    r.detection_rules/confidences/dates. The Cypher coalesces them to
    one-element arrays built from the legacy summary fields."""
    c, _ = client
    driver = _stub_session(
        [
            [
                {
                    "confidence": 0.92,
                    "rule_name": "exact_name_country_match_authority",
                    "detected_at": "2026-04-15T08:00:00Z",
                    # Server-side coalesce simulated: legacy edge with no
                    # parallel arrays returns single-element lists.
                    "det_rules": ["exact_name_country_match_authority"],
                    "det_confs": [0.92],
                    "det_dates": ["2026-04-15T08:00:00Z"],
                    "conflict": False,
                    # Why a pair is contested, not just that it is. The
                    # route surfaces these so a reviewer can see which
                    # identifier disagreed.
                    "conflict_property": None,
                    "conflict_left": None,
                    "conflict_right": None,
                    "a_labels": ["Authority"],
                    "a_props": {"authority_id": "AUTH-A", "name": "X"},
                    "b_labels": ["Authority"],
                    "b_props": {"authority_id": "AUTH-B", "name": "X"},
                }
            ]
        ]
    )
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)):
        r = c.get("/candidates?reviewed=false")
    body = r.json()
    assert len(body[0]["detections"]) == 1
    assert body[0]["detections"][0]["rule_name"] == "exact_name_country_match_authority"


def test_decide_not_found_returns_404(client):
    c, _ = client
    driver = _stub_session([[None]])
    with patch("src.api.routes.candidates.get_driver", AsyncMock(return_value=driver)):
        r = c.post(
            "/candidates/gmr-X/gmr-Y/decide",
            json={"decision": "reject", "reviewer": "alice"},
        )
    assert r.status_code == 404


# ── /decisions ────────────────────────────────────────────────────────────
# This module claimed to cover /decisions in its docstring but never did:
# the route sat at 19.4% line coverage, which is what held fontem-consolidator
# under the 80% gate. The filters below are the whole point of the endpoint —
# each one appends a WHERE clause and a bind, and none of that was exercised.


def _decisions_patch(rows):
    return patch("src.api.routes.decisions.audit.list_decisions", AsyncMock(return_value=rows))


def test_list_decisions_empty(client):
    c, _ = client
    with _decisions_patch([]):
        r = c.get("/decisions")
    assert r.status_code == 200
    assert r.json() == {"decisions": [], "next_cursor": None}


def test_list_decisions_passes_every_filter_to_the_log(client):
    """Each query param reaches the Postgres decision log as a filter."""
    c, _ = client
    with _decisions_patch([]) as listed:
        r = c.get(
            "/decisions?entity_type=Company&entity_id=abc&rule_name=fuzzy"
            "&decision_type=merge&since=2026-01-01&cursor=2026-04-01&limit=25"
        )
    assert r.status_code == 200
    assert listed.await_args.kwargs == {
        "entity_type": "Company", "entity_id": "abc", "rule_name": "fuzzy",
        "decision_type": "merge", "since": "2026-01-01", "cursor": "2026-04-01",
        "limit": 25,
    }


def test_list_decisions_sets_next_cursor_when_page_is_full(client):
    """A full page means there may be more, so hand back a cursor."""
    c, _ = client
    rows = [{"decided_at": f"2026-04-0{i}T00:00:00+00:00"} for i in (3, 2)]
    with _decisions_patch(rows):
        r = c.get("/decisions?limit=2")
    body = r.json()
    assert len(body["decisions"]) == 2
    assert body["next_cursor"] == "2026-04-02T00:00:00+00:00"


def test_list_decisions_omits_next_cursor_on_a_partial_page(client):
    """Fewer rows than the limit means the end; a cursor would loop forever."""
    c, _ = client
    with _decisions_patch([{"decided_at": "2026-04-03T00:00:00+00:00"}]):
        r = c.get("/decisions?limit=50")
    assert r.json()["next_cursor"] is None


def test_list_decisions_rejects_an_oversized_limit(client):
    """limit is capped at 1000 by the route signature."""
    c, _ = client
    r = c.get("/decisions?limit=5000")
    assert r.status_code == 422
