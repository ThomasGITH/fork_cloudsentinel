# Keep the CRCA runtime aligned with the other Python 3.12 services.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ARG PYTORCH_VERSION=2.2.2

# Set the working directory
WORKDIR /app

# Copy the runtime requirements separately to retain Docker layer caching.
COPY anomaly_detection/crca/requirements.txt /tmp/requirements.txt

# Torch is installed explicitly from the CPU wheel index. Keeping it out of the
# generic requirements file prevents pip from resolving CUDA/NVIDIA packages.
RUN python -m pip install --no-cache-dir --upgrade \
        pip==24.0 setuptools==69.2.0 wheel==0.43.0 \
    && python -m pip install --no-cache-dir \
        -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${PYTORCH_VERSION}+cpu" \
    && rm -f /tmp/requirements.txt

# Copy the rest of the application code into the container
COPY anomaly_detection/crca/ .
COPY data_ingestion/monitoring_history.py /app/monitoring_history.py

RUN groupadd --system --gid 10001 cloudsentinel \
    && useradd --system --uid 10001 --gid cloudsentinel --home-dir /app cloudsentinel \
    && mkdir -p /app/storage/monitoring_history /app/results \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 5023
# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5023"]
