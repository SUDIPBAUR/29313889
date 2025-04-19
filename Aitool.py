import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import os

# Binance setup
api_key = os.getenv("JiV7ENbyO4eDcrY7xi9eLNsKLWl8kL6JdtqOwH3O54MjxKLs004NvKAmSfcF0Huw")
api_secret = os.getenv("eTx3JFuXMbaCFPYJbAy4ZpYzeBW2eqKSDocldFqCDk8mv3xLT88fzB0IYARZLCjH")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})

symbol = 'BTC/USDT'
timeframe = '1h'

# Fetch historical data
def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Feature engineering (indicators, etc.)
def add_indicators(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['macd'] = MACD(df['close']).macd()
    df['ema'] = EMAIndicator(df['close']).ema_indicator()
    df['stoch'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
    bb = BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df.dropna(inplace=True)
    return df

# Prepare data for LSTM
def prepare_data(df):
    df['target'] = df['close'].shift(-1)
    features = ['close', 'rsi', 'macd', 'ema', 'stoch', 'cci', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'obv']
    data = df[features].values
    target = df['target'].values

    sequence_length = 20
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(1 if target[i+sequence_length] > data[i+sequence_length-1][0] else 0)

    X, y = np.array(X), np.array(y)
    return X, y

# Build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Inference and signal generation
def train_and_predict():
    df = fetch_data()
    df = add_indicators(df)
    X, y = prepare_data(df)
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X[:-10], y[:-10], epochs=5, batch_size=32, verbose=0)
    pred = model.predict(X[-1:])[0][0]
    signal = 'BUY' if pred > 0.6 else 'SELL' if pred < 0.4 else 'HOLD'
    confidence = pred
    timestamp = datetime.datetime.now()
    price = df['close'].iloc[-1]

    # Insert the signal into the database
    insert_signal(timestamp, signal, confidence, price)

    return df, signal, confidence

# Streamlit UI
st.title("BTC/USDT AI Signal Generator + Auto Trader")
df, signal, confidence = train_and_predict()

# Displaying the signal and price
st.line_chart(df['close'], height=300)
st.markdown(f"### Signal: **{signal}**")
st.markdown(f"Confidence Score: `{confidence:.2f}`")
st.dataframe(df.tail(10))

# Execute trade if not in paper mode
if st.button("Execute Trade"):
    result = place_order(signal)
    st.success(result)

# Signal history
def display_signal_history():
    # Fetch the signals from the database
    signals = fetch_signals()

    # Convert the result into a DataFrame for easier display
    signal_df = pd.DataFrame(signals, columns=['ID', 'Timestamp', 'Signal', 'Confidence', 'Price'])
    signal_df['Timestamp'] = pd.to_datetime(signal_df['Timestamp'])  # Convert timestamp to datetime

    # Display in Streamlit
    st.subheader("Signal History")
    st.write(signal_df)

# Display the signal history
display_signal_history()
