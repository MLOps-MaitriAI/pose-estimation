import mlflow
import dagshub
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Initialize Dagshub and MLflow
dagshub.init(repo_owner='MLOps-MaitriAI', repo_name='pose-estimation', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/MLOps-MaitriAI/pose-estimation.mlflow")

# Set the experiment name
experiment_name = "PE_SVC"
mlflow.set_experiment(experiment_name)

# Load the dataset
file_path = 'data/pose_landmarks_with_categories (3).csv'
data = pd.read_csv(file_path)

# Split data into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Support Vector Classifier
clf = SVC(kernel='rbf', random_state=42)

# Train the classifier
clf.fit(X_train_scaled, y_train)

# Save the trained model and scaler
model_path = './models/svc_model.sav'
scaler_path = './models/scaler.sav'
joblib.dump(clf, model_path)
joblib.dump(scaler, scaler_path)

# Load the pre-trained model and scaler
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Function to predict using the trained model
def predict(X):
    features_scaled = loaded_scaler.transform(X)
    predictions = loaded_model.predict(features_scaled)
    return predictions

# Start a new MLflow run
with mlflow.start_run(run_name="Pose_Estimation_Run_SVC"):
    # Log the dataset as an artifact
    mlflow.log_artifact(file_path, artifact_path="data")

    # Perform prediction on the test set
    predictions = predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Log parameters and metrics
    mlflow.log_param("model_path", model_path)
    mlflow.log_param("scaler_path", scaler_path)
    mlflow.log_param("kernel", clf.kernel)
    mlflow.log_metric("accuracy", accuracy)
    
    # Generate and log a confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save and log confusion matrix plot
    plot_filename = "confusion_matrix_svc.png"
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)

    # Log the model itself
    mlflow.sklearn.log_model(loaded_model, "pose_estimation_svc_model")

    # Cleanup
    os.remove(plot_filename)

# Explicitly end the MLflow run
mlflow.end_run()
