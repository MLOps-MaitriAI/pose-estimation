# Stage 1: Build Stage
FROM python:3.12-slim AS builder

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code into the container
COPY . .

# Stage 2: Runtime Stage
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=builder /app/best_model.sav /app/best_model.sav
COPY --from=builder /app/entrypoint.py /app/entrypoint.py
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Install only the runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point for the Docker container
ENTRYPOINT ["python", "/app/entrypoint.py"]
