FROM python:3.12-slim

WORKDIR /app
COPY anomaly_detection/generic/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY detectors /app/detectors
COPY learning_adaptation /app/learning_adaptation
COPY anomaly_detection /app/anomaly_detection
RUN groupadd --gid 10001 cloudsentinel && useradd --uid 10001 --gid 10001 --no-create-home cloudsentinel
USER 10001:10001
EXPOSE 5015
CMD ["python", "-m", "anomaly_detection.generic.app"]
