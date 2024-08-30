# Stage 1: Build
FROM python:3.12-slim AS builder

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Stage 2: Final runtime image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local /usr/local

# Copy the best model and entrypoint script into the container
COPY best_model.sav entrypoint.py ./

# Set the entry point for the Docker container
ENTRYPOINT ["python", "entrypoint.py"]
