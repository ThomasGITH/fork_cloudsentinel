# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY --chown=10001:10001 . .

RUN groupadd --system --gid 10001 cloudsentinel \
    && useradd --system --uid 10001 --gid cloudsentinel --home-dir /app cloudsentinel \
    && mkdir -p /app/storage/catalogue \
    && chown -R 10001:10001 /app

USER 10001:10001

EXPOSE 5001

# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
