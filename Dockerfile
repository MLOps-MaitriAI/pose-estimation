# Use a base image with Python
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the best model and the application code into the container
COPY models/best_model/best_model.sav /app/best_model/
COPY predict.py /app/

# Run the application
ENTRYPOINT ["python", "predict.py"]
