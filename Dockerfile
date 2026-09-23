# ── build: venv + void42 CA folded into the trust bundle ─────────────────────
FROM cgr.void42.internal/chainguard/python:latest-dev@sha256:0b39915a883dc2dbbe6c43be2ef82a37126409818cb66f5919e84cfa3e394ea5 AS build
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
FROM cgr.void42.internal/chainguard/python:latest@sha256:46e5b974e33be50d512688480df92f0e258ea849a562d9e63b701f8b080ea660
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
