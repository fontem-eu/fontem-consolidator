# ── build: venv + void42 CA folded into the trust bundle ─────────────────────
FROM cgr.void42.internal/chainguard/python:latest-dev@sha256:8af5085c793a9b501253117ccceabff2340400f3ef92fb0e09df690dd1e961a4 AS build
USER root
ENV PIP_INDEX_URL=https://nexus.void42.internal/repository/pypi-proxy/simple/ \
    PIP_TRUSTED_HOST=nexus.void42.internal
COPY void42-ca.crt /tmp/void42-ca.crt
RUN cat /tmp/void42-ca.crt >> /etc/ssl/certs/ca-certificates.crt
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vendor/gmr-event-schemas/ /tmp/gmr-event-schemas/
COPY vendor/gmr-events/        /tmp/gmr-events/
RUN pip install --no-cache-dir /tmp/gmr-event-schemas /tmp/gmr-events

# ── runtime: distroless; combined CA bundle so internal HTTPS is trusted ──────
FROM cgr.void42.internal/chainguard/python:latest@sha256:1206ffee8644e6338b3fc8b6e5dc384b03d91ad1df1d6b74fa4255544ac51ad2
WORKDIR /app
COPY --from=build /venv /venv
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
COPY src/ ./src/
USER 65532
EXPOSE 8000
ENTRYPOINT ["/venv/bin/uvicorn"]
CMD ["src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
