# ── build: venv + void42 CA folded into the trust bundle ─────────────────────
FROM cgr.void42.internal/chainguard/python:latest-dev@sha256:e7f69a8bf7fcae4010680cc2e6c28abf0cd172cfdf71d79cac29571a46b71da8 AS build
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
# The runtime needs the packages in the venv, not the tool that installed
# them: pip in a runtime image fetches and installs code (docker-build-sign
# checks runtime images for it).
RUN pip uninstall -y pip

# ── runtime: distroless; combined CA bundle so internal HTTPS is trusted ──────
FROM cgr.void42.internal/chainguard/python:latest@sha256:275d5493a184cf00a421871b1197de8eeb30c91280a8a18c222d61d9fb2c7919
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
