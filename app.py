import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("FinSight")

option = st.selectbox("Select Module", ["Credit Risk Prediction", "Stock Price Forecast"])

if option == "Credit Risk Prediction":
    st.header("Upload Credit Data for Default Prediction")
    file = st.file_uploader("Upload CSV file", type="csv")

    if file:
        try:
            # Load uploaded CSV, skip extra header row if exists
            df = pd.read_csv(file, skiprows=1)

            # Drop 'Unnamed: 0' if present
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])

            st.write("Uploaded Data Preview:")
            st.dataframe(df.head())

            # Load retrained model and scaler
            model = joblib.load("models/credit_model.pkl")
            scaler = joblib.load("models/scaler.pkl")

            # Scale features
            df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

            # Predict using the model
            predictions = model.predict(df_scaled)
            df['Prediction'] = predictions

            st.write("Predictions:")
            st.dataframe(df[['Prediction']])

            # Save predictions to CSV and provide download button
            output_path = "outputs/predictions.csv"
            df.to_csv(output_path, index=False)

            st.success(f"Predictions saved to {output_path}")
            st.download_button(
                label="Download Predictions CSV",
                data=open(output_path, "rb").read(),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing the file: {e}")

elif option == "Stock Price Forecast":
    st.header("Enter Stock Ticker for Price Forecast")
    ticker = st.text_input("Ticker symbol (e.g. AAPL)")

    if ticker:
        data = yf.download(ticker, start="2022-01-01", end="2023-01-01")
        if data.empty:
            st.error("No data found for ticker. Please try another.")
        else:
            st.line_chart(data['Close'])

            # Simple Linear Regression Forecast
            data = data.dropna()
            data['day'] = np.arange(len(data))
            X = data['day'].values.reshape(-1, 1)
            y = data['Close'].values

            model = LinearRegression()
            model.fit(X, y)
            forecast = model.predict(X)

            st.line_chart(pd.DataFrame({'Actual': y, 'Forecast': forecast}))
            st.success("Forecast plotted above.")
