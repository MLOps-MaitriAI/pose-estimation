# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the required files into the container
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model directory
COPY models /app/models

# Copy the prediction script
COPY predict.py /app/predict.py

# Command to run the prediction script
ENTRYPOINT ["python", "predict.py"]
