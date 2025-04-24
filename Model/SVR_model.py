import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
energy = pd.read_csv("energy_data.csv")

# Define features and target
X = energy.drop(columns=['Germany/Luxembourg euro per MWh Original resolutions', "MESS_DATUM"])
y = energy['Germany/Luxembourg euro per MWh Original resolutions']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Move data to GPU using PyTorch (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVR (still on CPU, because sklearn's SVR doesn't support CUDA)
svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
svr.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = svr.predict(X_train_scaled)
y_pred_test = svr.predict(X_test_scaled)

# Move predictions to GPU
y_pred_train_tensor = torch.tensor(y_pred_train, dtype=torch.float32, device=device)
y_pred_test_tensor = torch.tensor(y_pred_test, dtype=torch.float32, device=device)

# Compute evaluation metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Print results
print(f"Train Score (R²): {train_r2:.4f}")
print(f"Test Score (R²): {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Print GPU predictions
print("Train Predictions (GPU Tensor):", y_pred_train_tensor)
print("Test Predictions (GPU Tensor):", y_pred_test_tensor)
