import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib  # For saving the model

# Load dataset
energy = pd.read_csv("new_energy_data.csv")

# Define features and target
X = energy.drop(columns=['Germany/Luxembourg euro per MWh Original resolutions', "MESS_DATUM"])
y = energy['Germany/Luxembourg euro per MWh Original resolutions']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# Create the XGBoost model
xgb = XGBRegressor(objective='reg:squarederror', tree_method='auto', random_state=42)

# Hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [400,500],
    'learning_rate': [0.01,0.05],
    'max_depth': [4,5],
'lambda': [1,2],
    'alpha': [1,2]
}

# Use GridSearchCV with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters from grid search
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Get the best XGBoost model from grid search
best_xgb = grid_search.best_estimator_

# Train the best XGBoost model on the entire training set
best_xgb.fit(X_train_scaled, y_train)

# Save the best model using joblib
model_filename = 'best_xgb_model.pkl'
joblib.dump(best_xgb, model_filename)
print(f"Best model saved as {model_filename}")

# Make predictions on training and testing set
y_pred_train = best_xgb.predict(X_train_scaled)
y_pred_test = best_xgb.predict(X_test_scaled)

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

# Print predictions
print("Train Predictions:", y_pred_train)
print("Test Predictions:", y_pred_test)

# Move predictions to GPU (if necessary)
y_pred_train_tensor = torch.tensor(y_pred_train, dtype=torch.float32, device=device)
y_pred_test_tensor = torch.tensor(y_pred_test, dtype=torch.float32, device=device)

print("Train Predictions (GPU Tensor):", y_pred_train_tensor)
print("Test Predictions (GPU Tensor):", y_pred_test_tensor)
