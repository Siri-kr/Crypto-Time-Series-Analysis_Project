import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from models.evaluation import evaluate_all_models

# Sidebar
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Price Chart", "Moving Average", "LSTM Prediction", "Model Comparison"]
)

# Load data
df = pd.read_csv("data/cleaned_btc.csv")

# ------------------ OVERVIEW ------------------
if page == "Overview":
    st.title("Crypto Analytics Dashboard 🚀")
    st.write(df.head())

# ------------------ PRICE CHART ------------------
elif page == "Price Chart":
    st.title("Bitcoin Price Chart 📈")

    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Close'])
    plt.xticks(rotation=45)
    st.pyplot(plt)

# ------------------ MOVING AVERAGE ------------------
elif page == "Moving Average":
    st.title("Moving Average 📊")

    df['MA50'] = df['Close'].rolling(50).mean()

    plt.figure(figsize=(10,5))
    plt.plot(df['Close'], label="Price")
    plt.plot(df['MA50'], label="MA50")
    plt.legend()
    st.pyplot(plt)

# ------------------ LSTM ------------------
elif page == "LSTM Prediction":
    st.title("LSTM Prediction 🔮")

    from models.lstm_model import run_lstm
    import matplotlib.pyplot as plt

    actual, predicted = run_lstm()

    st.subheader("Actual vs Predicted")

    plt.figure(figsize=(10,5))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.legend()

    st.pyplot(plt)
# ------------------ MODEL COMPARISON ------------------
elif page == "Model Comparison":
    st.subheader("Model Performance")

    results = evaluate_all_models()

    for model, rmse in results.items():
        st.write(f"{model} RMSE: {rmse:.2f}")

    best_model = min(results, key=results.get)
    st.success(f"{best_model} performs better ✅")