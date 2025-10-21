import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. 📥 Charger les données de NVDA
ticker = 'NVDA'
data = yf.download(ticker, start='2015-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
close_data = data[['Close']]

# 2. 🔍 Normalisation
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# 3. 🧱 Créer les séquences (60 jours pour prédire le 61e)
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Reshape pour LSTM: (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. 🧠 Construire le modèle LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# 5. 🔮 Prédire les 10 prochains jours
last_sequence = scaled_data[-sequence_length:]
future_predictions = []
current_seq = last_sequence.copy()

for _ in range(10):
    input_seq = np.reshape(current_seq, (1, sequence_length, 1))
    pred = model.predict(input_seq, verbose=0)
    future_predictions.append(pred[0, 0])
    current_seq = np.append(current_seq[1:], pred, axis=0)

# Inverser la normalisation
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 6. 📈 Visualisation
plt.figure(figsize=(12, 6))
plt.plot(close_data.index[-200:], close_data['Close'][-200:], label='Données réelles')
future_index = pd.date_range(start=close_data.index[-1] + pd.Timedelta(days=1), periods=10, freq='B')
plt.plot(future_index, future_predictions, label='Prévision (10 jours)', linestyle='--')
plt.title(f'Prévision du cours de NVDA avec LSTM')
plt.xlabel('Date')
plt.ylabel('Prix de clôture')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
