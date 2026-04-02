from sklearn.metrics import mean_squared_error
import numpy as np

from models.arima_model import run_arima
from models.prophet_model import run_prophet
from models.lstm_model import run_lstm


def evaluate_all_models():
    results = {}

    # ARIMA
    actual, pred = run_arima()
    results["ARIMA"] = np.sqrt(mean_squared_error(actual, pred))

    # Prophet
    actual, pred = run_prophet()
    results["Prophet"] = np.sqrt(mean_squared_error(actual, pred))

    # LSTM
    actual, pred = run_lstm()
    results["LSTM"] = np.sqrt(mean_squared_error(actual, pred))

    return results