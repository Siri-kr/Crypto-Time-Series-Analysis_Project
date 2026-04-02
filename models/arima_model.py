import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def run_arima():
    # Load data
    df = pd.read_csv("data/cleaned_btc.csv", parse_dates=['Date'], index_col='Date')

    # Train ARIMA model
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()

    # Split data (for evaluation)
    train = df['Close'][:-10]
    test = df['Close'][-10:]

    # Forecast
    forecast = model_fit.forecast(steps=10)

    # Return actual vs predicted
    return test.values, forecast.values