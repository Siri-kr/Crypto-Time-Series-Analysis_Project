import pandas as pd

# Read CSV by skipping first 3 rows (junk headers)
df = pd.read_csv("data/btc.csv", skiprows=3)

print("Original Data:")
print(df.head())

# Rename columns manually
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop invalid rows
df = df.dropna(subset=['Date'])

# Set Date as index
df.set_index('Date', inplace=True)

# Keep only Close column
df = df[['Close']]

# Fill missing values
df = df.ffill()

# Save cleaned data
df.to_csv("data/cleaned_btc.csv")

print("\n✅ Cleaned Data:")
print(df.head())