import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Download historical data for Apple (AAPL)
ticker = 'AAPL'
data = yf.download(ticker, start='2022-01-01', end='2023-01-01')
data = data[['Close']].dropna()

# Prepare data for regression
data['Day'] = np.arange(len(data))
X = data['Day'].values.reshape(-1, 1)
y = data['Close'].values

# Train linear regression model
model = LinearRegression()
model.fit(X, y)
forecast = model.predict(X)

# Plot actual vs forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, y, label='Actual Close Price')
plt.plot(data.index, forecast, label='Forecasted Price', linestyle='--')
plt.title(f'Stock Price Forecast for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
