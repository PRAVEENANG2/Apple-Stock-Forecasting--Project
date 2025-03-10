import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

# Load the trained GRU model
gru_model = load_model('gru_model.h5')

# Load your dataset
apple_data = pd.read_csv('apple_stock.csv', parse_dates=['Date'], index_col='Date')

# Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(apple_data[['Close']])
test_scaled = data_scaled[-30:]  # Last 30 days data

# Generate future predictions
future_days = 30
last_value = test_scaled[-1].reshape(1,1,1)
predictions = []
dates = [apple_data.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]

for _ in range(future_days):
    predicted_value = gru_model.predict(last_value)
    predictions.append(predicted_value[0][0])
    last_value = np.array(predicted_value).reshape(1,1,1)

# Convert predictions back to original scale
future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
future_prices = np.clip(future_prices, apple_data['Close'].min(), apple_data['Close'].max())

# Create DataFrame for visualization
forecast_df = pd.DataFrame({'Date': dates, 'Predicted_Close': future_prices.flatten()})
forecast_df.set_index('Date', inplace=True)

# Streamlit UI
st.title('Apple Stock Price Forecast')
st.write('Forecast for the next 30 days using GRU model')

# Plot results
st.subheader('Stock Price Forecast')
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(apple_data.index, apple_data['Close'], label='Historical Prices', color='blue')
ax.plot(forecast_df.index, forecast_df['Predicted_Close'], label='GRU Forecast', color='red', linestyle='dashed')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Display forecasted values
st.subheader('Forecasted Prices')
st.dataframe(forecast_df)
