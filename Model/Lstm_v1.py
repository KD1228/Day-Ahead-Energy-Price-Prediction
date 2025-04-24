# -*- coding: utf-8 -*-
"""
Energy Price Prediction with Corrected Scaling and Simplified Architecture
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model

# Custom R² metric function
def r_squared(y_true, y_pred):
    ss_res = keras.backend.sum(keras.backend.square(y_true - y_pred))
    ss_tot = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))
    return 1 - ss_res/(ss_tot + keras.backend.epsilon())

# Data loading and preprocessing
def load_and_preprocess_data(filepath):
    energy = pd.read_csv(filepath)
    energy["MESS_DATUM"] = pd.to_datetime(energy["MESS_DATUM"])
    energy = energy.sort_values("MESS_DATUM")
    
    # Feature engineering
    def cyclical_encode(df, col, max_val):
        df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
        return df

    # Temporal features
    temporal_features = {
        'Hour': 24,
        'Day': 31,
        'Month': 12,
        'Weekday': 7
    }
    
    for col, max_val in temporal_features.items():
        energy[col] = getattr(energy['MESS_DATUM'].dt, col.lower())
        energy = cyclical_encode(energy, col, max_val)

    # Energy features
    energy["Total_Nonrenewable"] = energy[[
        'Lignite [MWh] Calculated resolutions', 
        'Hard coal [MWh] Calculated resolutions',
        'Fossil gas [MWh] Calculated resolutions', 
        'Other conventional [MWh] Calculated resolutions'
    ]].sum(axis=1)

    # Lag features
    for lag in [24, 168]:  # 24h and weekly lags
        energy[f'lag_{lag}'] = energy['Total_Nonrenewable'].shift(lag)

    # Drop unnecessary columns
    energy.drop(columns=[
        'MESS_DATUM', 'Lignite [MWh] Calculated resolutions',
        'Hard coal [MWh] Calculated resolutions', 'Fossil gas [MWh] Calculated resolutions',
        'Other conventional [MWh] Calculated resolutions', 'Biomass [MWh] Calculated resolutions',
        'Hydropower [MWh] Calculated resolutions', 'Wind offshore [MWh] Calculated resolutions',
        'Wind onshore [MWh] Calculated resolutions', 'Photovoltaics [MWh] Calculated resolutions',
        'Other renewable [MWh] Calculated resolutions'
    ], inplace=True)
    
    return energy.dropna()

# Sequence creation function
def create_sequences(data, seq_length, horizon):
    x, y = [], []
    for i in range(len(data) - seq_length - horizon):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+horizon, 0])  # First column is target
    return np.array(x), np.array(y)

# Model architecture
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(128, return_sequences=True, dropout=0.3)(inputs)
    
    # Second LSTM layer
    x = LSTM(64, dropout=0.3)(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(forecast_horizon)(x)
    
    return Model(inputs, outputs)

# Main execution
if __name__ == "__main__":
    # Parameters
    file_path = "energy_data.csv"
    sequence_length = 168  # 1 week of hourly data
    forecast_horizon = 24   # Predict next 24 hours
    
    # Load and preprocess data
    energy = load_and_preprocess_data(file_path)
    data = energy.to_numpy().astype('float32')
    
    # Create sequences
    X, y = create_sequences(data, sequence_length, forecast_horizon)
    
    # Train/val/test split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    # Feature scaling (only on features, not target)
    feature_scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = feature_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Get target scaling parameters
    target_scaler_mean = feature_scaler.mean_[0]
    target_scaler_scale = feature_scaler.scale_[0]

    # Build and compile model
    model = build_model((sequence_length, X_train.shape[-1]))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='huber',
        metrics=[r_squared, 'mae']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
    ]

    # Training
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=150,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    # Load best model
    best_model = keras.models.load_model('best_model.keras', custom_objects={'r_squared': r_squared})

    # Evaluation
    y_pred = best_model.predict(X_test_scaled)
    
    # Inverse scaling for target variable
    y_test_original = y_test * target_scaler_scale + target_scaler_mean
    y_pred_original = y_pred * target_scaler_scale + target_scaler_mean

    # Metrics
    print(f'\nTest R²: {r2_score(y_test_original, y_pred_original):.3f}')
    print(f'Test MAE: {mean_absolute_error(y_test_original, y_pred_original):.2f}')

    # Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_original[-100:].flatten(), label='True')
    plt.plot(y_pred_original[-100:].flatten(), label='Predicted')
    plt.title('Last 100 Hours Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Error analysis
    errors = y_pred_original - y_test_original
    plt.figure(figsize=(10, 6))
    sns.histplot(errors.flatten(), kde=True, bins=50)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.show()