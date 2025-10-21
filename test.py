import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from datetime import timedelta
import pytz
from pytz import timezone

# 1. Télécharger les données NVDA à la minute sur 7 jours
df = yf.download("NVDA", interval="1m", period="7d")[['Close']].dropna()

# 2. Calcul EMA 10 et RSI 14
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))
df.dropna(inplace=True)

# 3. Normalisation
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 4. Séquences (X, y)
X, y = [], []
seq_len = 60
for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i-seq_len:i])
    y.append(scaled_data[i, 0])  # Close

X, y = np.array(X), np.array(y)

# 5. Modèle LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
model.fit(X, y, epochs=5, batch_size=64, verbose=1)

# 6. Prédiction 15 prochaines minutes
last_seq = scaled_data[-seq_len:]
predictions = []

for _ in range(15):
    pred = model.predict(np.expand_dims(last_seq, axis=0), verbose=0)[0][0]
    new_point = [pred] + list(last_seq[-1][1:])  # Conserver EMA & RSI
    last_seq = np.append(last_seq[1:], [new_point], axis=0)
    predictions.append(pred)

# 7. Dénormalisation
dummy = np.zeros((15, scaled_data.shape[1]))
dummy[:, 0] = predictions  # Uniquement la colonne 'Close'
predicted_prices = scaler.inverse_transform(dummy)[:, 0]

# 8. Heure de Paris (en temps réel)
start = pd.Timestamp.now(tz=timezone("Europe/Paris")).floor("min")

print("\n📈 Prédictions NVDA pour les 15 prochaines minutes (heure de Paris) :")
for i, price in enumerate(predicted_prices):
    heure = (start + timedelta(minutes=i+1)).strftime("%Hh%M")
    print(f"⏰ À {heure}, NVDA ≈ {price:.2f} $")
