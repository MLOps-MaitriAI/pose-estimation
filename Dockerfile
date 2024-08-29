# Use a base image with Python
FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC and other dependencies
RUN pip install dvc dagshub mlflow

# Copy the entire project
COPY . .

# Expose the necessary port (adjust if needed)
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "app3.py"]
