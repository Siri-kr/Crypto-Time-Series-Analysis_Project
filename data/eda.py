import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("data/cleaned_btc.csv", parse_dates=['Date'], index_col='Date')

print(df.head())

# 1. Price Trend
plt.figure(figsize=(10,5))
plt.plot(df['Close'])
plt.title("Bitcoin Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.show()

# 2. Moving Average
df['MA20'] = df['Close'].rolling(window=20).mean()

plt.figure(figsize=(10,5))
plt.plot(df['Close'], label="Close Price")
plt.plot(df['MA20'], label="MA20", color='red')
plt.title("Moving Average")
plt.legend()
plt.grid()
plt.show()