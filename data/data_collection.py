import yfinance as yf

# Download data properly
df = yf.download("BTC-USD", period="1y", interval="1d")

# Check if data is empty
if df.empty:
    print("❌ Data not downloaded. Try again.")
else:
    df.to_csv("data/btc.csv")
    print("✅ Data downloaded successfully!")
    print(df.head())