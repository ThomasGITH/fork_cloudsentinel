# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Build this image from the repository root:
# docker build -f learning_adaptation/Dockerfile .
COPY learning_adaptation/requirements.txt ./requirements.txt

# Install the required packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install the shared detector registry and its manifest data.
COPY pyproject.toml ./shared/pyproject.toml
COPY detectors ./shared/detectors
RUN pip install --no-cache-dir ./shared

# Copy the rest of the application code into the container
COPY --chown=10001:10001 learning_adaptation .

# Fail the image build if the API and worker would contain the registered
# comparison task without its runtime modules.
RUN test -f /app/comparison_api.py \
    && test -f /app/comparison_execution.py \
    && test -f /app/comparison_evaluation.py \
    && test -f /app/comparison_storage.py \
    && python -c "import comparison_api, comparison_evaluation, comparison_execution, comparison_storage"

RUN groupadd --system --gid 10001 cloudsentinel \
    && useradd --system --uid 10001 --gid cloudsentinel --home-dir /app cloudsentinel \
    && mkdir -p /app/storage/dataset_snapshots \
        /app/storage/training_runs \
        /app/storage/trained_models_temp \
        /app/storage/models \
        /app/storage/model_artifacts \
        /app/storage/comparisons \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 5005

# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5005"]
