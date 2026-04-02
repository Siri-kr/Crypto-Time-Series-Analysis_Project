import pandas as pd
from prophet import Prophet

def run_prophet():   # ✅ IMPORTANT FUNCTION

    # Load data
    df = pd.read_csv("data/cleaned_btc.csv")

    # Prophet format
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']

    # Train model
    model = Prophet()
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=10)

    # Predict
    forecast = model.predict(future)

    # Get last 10 actual & predicted
    actual = df['y'].values[-10:]
    predicted = forecast['yhat'].values[-10:]

    return actual, predicted   # ✅ RETURN