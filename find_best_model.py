import os

# Locate the best model
model_folder = './models/best_model/'
model_files = os.listdir(model_folder)

if model_files:
    latest_model = max(model_files, key=lambda f: os.path.getctime(os.path.join(model_folder, f)))
    print(latest_model)
else:
    raise Exception('No model found in the best_model folder')
