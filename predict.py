import joblib
import os
import numpy as np

# Load the model
model_path = './best_model/best_model.sav'
model = joblib.load(model_path)

def predict(input_data):
    # Example prediction function
    return model.predict(input_data)

# Example usage
if __name__ == "__main__":
    # Example input data (replace with actual data)
    input_data = np.array([[0.1, 0.2, 0.3]])  # Adjust dimensions as needed
    print(predict(input_data))
