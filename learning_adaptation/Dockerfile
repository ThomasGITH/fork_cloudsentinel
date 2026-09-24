# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

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
COPY learning_adaptation .

EXPOSE 5005

# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5005"]
