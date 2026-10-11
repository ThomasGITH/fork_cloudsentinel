# Use the official Python image from the Docker Hub
FROM python:3.12

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY anomaly_detection/crca/requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir scikit_network==0.31.0

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
