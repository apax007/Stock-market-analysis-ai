code = """
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # Technical analysis library
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Fetch Stock Data
def fetch_stock_data(symbol, period="1y"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    if data.empty:
        st.warning(f"No data found for {symbol}")
        return None
    return data

# Apply Technical Indicators
def apply_indicators(data):
    data["SMA_20"] = ta.trend.sma_indicator(data["Close"], window=20)
    data["SMA_50"] = ta.trend.sma_indicator(data["Close"], window=50)
    data["RSI"] = ta.momentum.rsi(data["Close"], window=14)
    data["MACD"] = ta.trend.macd(data["Close"])
    return data

# Prepare Data for AI Model
def prepare_data_for_ai(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

    X_train, y_train = [], []
    for i in range(50, len(scaled_data)):
        X_train.append(scaled_data[i-50:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler

# Train LSTM Model
def train_lstm_model(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, batch_size=1, epochs=3, verbose=1)
    return model

# Predict Stock Prices
def predict_stock_trend(model, data, scaler):
    scaled_data = scaler.transform(data["Close"].values.reshape(-1,1))

    X_test = []
    for i in range(50, len(scaled_data)):
        X_test.append(scaled_data[i-50:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    data["Predicted"] = np.nan
    data.iloc[50:, data.columns.get_loc("Predicted")] = predictions.flatten()

    return data

# Streamlit Web UI
def main():
    st.title("📈 AI Stock Market Analysis Tool")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
    period = st.selectbox("Select Timeframe", ["1mo", "3mo", "6mo", "1y", "5y"])

    if st.button("Analyze Stock"):
        data = fetch_stock_data(stock_symbol, period)
        if data is not None:
            data = apply_indicators(data)

            st.subheader("Stock Data Overview")
            st.write(data.tail())

            st.subheader("Stock Price & Indicators")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(data.index, data["Close"], label="Close Price", color="cyan")
            ax.plot(data.index, data["SMA_20"], label="SMA 20", color="orange", linestyle="--")
            ax.plot(data.index, data["SMA_50"], label="SMA 50", color="red", linestyle="--")
            plt.legend()
            st.pyplot(fig)

            # AI Predictions
            X_train, y_train, scaler = prepare_data_for_ai(data)
            model = train_lstm_model(X_train, y_train)
            data = predict_stock_trend(model, data, scaler)

            st.subheader("📊 AI Predicted Trends")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(data.index, data["Close"], label="Close Price", color="cyan")
            ax.plot(data.index, data["Predicted"], label="Predicted Price", color="magenta")
            plt.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main():
"""
with open("app.py", "w") as file:
    file.write(code)
