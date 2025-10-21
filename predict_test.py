import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Télécharger les données NVDA (intervalle 1h)
df = yf.download("NVDA", interval="1h", period="30d")
df = df[['Close']].dropna()

# 2. Normaliser les données
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 3. Créer les séquences (X, y)
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# 4. Séparer en train/test
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 5. Construire le modèle LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# 7. Prédire le prix à la prochaine heure
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)
predicted_price_scaled = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

print(f"💰 Prédiction du prix NVDA pour la prochaine heure : {predicted_price:.2f} $")
