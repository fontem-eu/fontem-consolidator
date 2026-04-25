"""Route-level unit tests for /candidates and /decisions."""

from unittest.mock import AsyncMock, MagicMock, patch


def _stub_session(run_results):
    session = AsyncMock()

    class _Result:
        def __init__(self, recs):
            self._recs = recs

        def __aiter__(self):
            self._it = iter(self._recs)
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

        async def single(self):
            return self._recs[0] if self._recs else None

    async def run(*args, **kwargs):
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
                    "detections": [
                        {
                            "rule_name": "fuzzy_name_same_country",
                            "confidence": 0.94,
                            "detected_at": "2026-04-20T12:00:00Z",
                        },
                        {
                            "rule_name": "embedding_cosine_authority",
                            "confidence": 0.87,
                            "detected_at": "2026-04-20T12:05:00Z",
                        },
                    ],
                    "conflict": False,
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
    """Edges written before the multi-rule schema have no r.detections.
    The Cypher coalesces a single-element list from the legacy summary."""
    c, _ = client
    driver = _stub_session(
        [
            [
                {
                    "confidence": 0.92,
                    "rule_name": "exact_name_country_match_authority",
                    "detected_at": "2026-04-15T08:00:00Z",
                    # Coalesced server-side; the route still receives a list.
                    "detections": [
                        {
                            "rule_name": "exact_name_country_match_authority",
                            "confidence": 0.92,
                            "detected_at": "2026-04-15T08:00:00Z",
                        },
                    ],
                    "conflict": False,
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
