# Use a base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install DVC, Dagshub, MLflow, and any other dependencies
RUN pip install dvc dagshub mlflow

# Copy the entire project into the container
COPY . .

# Expose the necessary port for Streamlit (default is 8501)
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "app3.py"]
