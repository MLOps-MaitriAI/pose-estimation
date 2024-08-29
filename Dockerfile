# Use a lightweight base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the dependencies file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and the rest of the application code
COPY models/best_model /app/models/best_model
COPY . /app/

# Set the command to run the model server (example with Flask)
CMD ["python", "app.py"]
