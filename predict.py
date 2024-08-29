import os
import joblib
import numpy as np

# Determine the best model path
best_model_name = os.getenv('BEST_MODEL')
model_path = f'./models/{best_model_name}'

if not os.path.isfile(model_path):
    raise FileNotFoundError(f'Model file {model_path} does not exist.')

# Load the model
model = joblib.load(model_path)

# Example prediction (modify as needed)
def make_predictions():
    # Dummy data for prediction
    X = np.random.rand(1, 10)  # Adjust shape as needed
    predictions = model.predict(X)
    print("Predictions:", predictions)

if __name__ == "__main__":
    make_predictions()
