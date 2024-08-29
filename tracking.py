import mlflow
import dagshub
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import shutil  # Import shutil for file operations

# Initialize Dagshub and MLflow
dagshub.init(repo_owner='MLOps-MaitriAI', repo_name='pose-estimation', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/MLOps-MaitriAI/pose-estimation.mlflow")

# Set the experiment name
experiment_name = "PE_Model_Comparison"
mlflow.set_experiment(experiment_name)

# File paths for each dataset
file_paths = [
    'data/pose_landmarks_with_categories (1).csv',
    'data/pose_landmarks_with_categories (2).csv',
    'data/pose_landmarks_with_categories (3).csv'
]

# Model configurations
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    "SVC": SVC(kernel='rbf', random_state=42)
}

# Store metrics for comparison
results = {}
model_paths = {}

for file_path in file_paths:
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Split data into features (X) and target (y)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Determine which model to use based on file name
    if '1' in file_path:
        model_name = "KNN"
    elif '2' in file_path:
        model_name = "RandomForest"
    elif '3' in file_path:
        model_name = "SVC"
    else:
        continue
    
    model = models[model_name]
    
    # Train the classifier
    model.fit(X_train_scaled, y_train)
    
    # Save the trained model and scaler
    model_path = f'./models/{model_name.lower()}_model.sav'
    scaler_path = './models/scaler.sav'
    joblib.dump(model, model_path)
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
    with mlflow.start_run(run_name=f"Pose_Estimation_Run_{model_name}"):
        # Log the dataset as an artifact
        mlflow.log_artifact(file_path, artifact_path="data")
        
        # Perform prediction on the test set
        predictions = predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        results[model_name] = accuracy
        model_paths[model_name] = model_path
        
        # Log parameters and metrics
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("scaler_path", scaler_path)
        if model_name == "KNN":
            mlflow.log_param("n_neighbors", model.n_neighbors)
        elif model_name == "RandomForest":
            mlflow.log_param("n_estimators", model.n_estimators)
        elif model_name == "SVC":
            mlflow.log_param("kernel", model.kernel)
        mlflow.log_metric("accuracy", accuracy)
        
        # Generate and log a confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {model_name}')
        
        # Save and log confusion matrix plot
        plot_filename = f"confusion_matrix_{model_name.lower()}.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        
        # Log the model itself
        mlflow.sklearn.log_model(loaded_model, f"pose_estimation_{model_name.lower()}_model")
        
        # Cleanup
        os.remove(plot_filename)
    
    # Explicitly end the MLflow run
    mlflow.end_run()

# Determine the best model
best_model = max(results, key=results.get)
print(f"The best model is: {best_model} with an accuracy of {results[best_model]}")

# Save the best model to a specific directory
best_model_path = model_paths[best_model]
best_model_dest_path = './models/best_model'
os.makedirs(best_model_dest_path, exist_ok=True)
shutil.copy(best_model_path, os.path.join(best_model_dest_path, 'best_model.sav'))

print(f"The best model has been saved to: {best_model_dest_path}/best_model.sav")
