FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    IF_MODEL_STORAGE_ROOT=/app/storage/trained_models \
    IF_RESULTS_STORAGE_ROOT=/app/storage/results \
    IF_MAX_UPLOAD_BYTES=104857600

WORKDIR /app

# Build from the repository root:
# docker build -f anomaly_detection/isolation_forest/Dockerfile \
#   -t anomaly_detection_isolation_forest .
COPY anomaly_detection/isolation_forest/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./shared/pyproject.toml
COPY detectors ./shared/detectors
RUN pip install --no-cache-dir ./shared

COPY anomaly_detection/isolation_forest ./anomaly_detection/isolation_forest

RUN groupadd --system --gid 10001 cloudsentinel && \
    useradd --system --uid 10001 --gid cloudsentinel --home-dir /app cloudsentinel && \
    mkdir -p /app/storage/trained_models /app/storage/results && \
    chown -R cloudsentinel:cloudsentinel /app

USER 10001:10001

EXPOSE 5014

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5014/healthz', timeout=2).read()" || exit 1

CMD ["python", "-m", "flask", "--app", "anomaly_detection.isolation_forest.app", "run", "--host=0.0.0.0", "--port=5014"]
