"""HTTP client for the gmr-linguistics service — embeddings.

Translation moved to fontem-translator; the consolidator only needs the
name vectors its similarity rule compares.

Thin async wrapper — no retry/breaker of its own. The service already does
retries + circuit-breaker + spend-cap in-process; adding a second layer here
would double the failure surface. Errors bubble up so the consolidator can
decide what to do (log + skip, retry later, etc).
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


class LinguisticsError(Exception):
    """Unrecoverable failure from gmr-linguistics."""


class LinguisticsUnavailable(Exception):
    """Transient: service returned 5xx/503 or timed out."""


@dataclass
class LinguisticsClient:
    base_url: str
    timeout_s: float = 60.0
    # Default matches settings.linguistics_embedding_backend — labse-local
    # is 768-d, consistent with the authority_name_embedding_idx vector
    # index. Callers that want another encoder must also provision an
    # index of its dimension.
    embedding_backend: str = "labse-local"
    _client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LinguisticsClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"), timeout=self.timeout_s,
        )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LinguisticsClient used outside `async with` block")
        return self._client

    async def embed(self, text: str) -> tuple[list[float], str]:
        """Return (vector, encoder_id).

        encoder_id is the signed-mirror identifier the linguistics service
        stamps on every response — consumers store it alongside the vector
        so that cross-encoder comparisons can be rejected downstream. A
        missing encoder_id from the service is treated as a hard error;
        we never write an un-versioned vector.
        """
        resp = await self._post_json("/embed", {
            "text": text, "backend": self.embedding_backend,
        })
        vec = resp.get("vector")
        if not isinstance(vec, list) or not vec:
            raise LinguisticsError("malformed /embed response: missing vector")
        encoder_id = resp.get("encoder_id")
        if not isinstance(encoder_id, str) or not encoder_id:
            raise LinguisticsError(
                "malformed /embed response: missing encoder_id — refusing "
                "to write an un-versioned embedding",
            )
        return [float(x) for x in vec], encoder_id

    async def _post_json(self, path: str, payload: dict) -> dict:
        try:
            r = await self.client.post(path, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LinguisticsUnavailable(str(exc)) from exc

        if r.status_code >= 500 or r.status_code == 429:
            # 429 = spend cap hit, 503 = circuit-open or backend unavailable.
            # Both are transient from the consolidator's POV — retry later.
            raise LinguisticsUnavailable(
                f"status={r.status_code} body={r.text[:200]}"
            )
        if r.status_code >= 400:
            raise LinguisticsError(
                f"status={r.status_code} body={r.text[:300]}"
            )
        return r.json()
