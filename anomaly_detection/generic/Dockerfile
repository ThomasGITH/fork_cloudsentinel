FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_CATALOGUE_STORAGE_ROOT=/app/learning-storage/models \
    MODEL_ARTIFACT_STORAGE_ROOT=/app/storage/model_artifacts \
    GENERIC_DETECTION_RESULTS_ROOT=/app/storage/results

WORKDIR /app

# Build from the repository root:
# docker build -f anomaly_detection/generic/Dockerfile \
#   -t jojojochem/anomaly_detection_generic:generic-detection-1 .
COPY anomaly_detection/generic/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY pyproject.toml /tmp/cloudsentinel-detectors/pyproject.toml
COPY detectors /tmp/cloudsentinel-detectors/detectors
RUN pip install --no-cache-dir /tmp/cloudsentinel-detectors \
    && rm -rf /tmp/cloudsentinel-detectors /tmp/requirements.txt

COPY --chown=10001:10001 learning_adaptation /app/learning_adaptation
COPY --chown=10001:10001 anomaly_detection/generic /app/anomaly_detection/generic

RUN groupadd --system --gid 10001 cloudsentinel \
    && useradd --system --uid 10001 --gid cloudsentinel --home-dir /app cloudsentinel \
    && mkdir -p /app/storage/model_artifacts /app/storage/results /app/learning-storage/models \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 5015

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5015/healthz', timeout=2).read()" || exit 1

CMD ["gunicorn", "--bind=0.0.0.0:5015", "--workers=1", "--threads=4", "--timeout=120", "anomaly_detection.generic.app:app"]
