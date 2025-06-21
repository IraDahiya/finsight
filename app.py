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
            # Load uploaded CSV
            df = pd.read_csv(file)

            # Drop unwanted columns if any (like 'Unnamed: 0')
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])

            # Required features to match model training
            required_features = [
                'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
            ]

            # Check if all required columns are present
            if not all(feature in df.columns for feature in required_features):
                missing = list(set(required_features) - set(df.columns))
                st.error(f"Missing required columns: {missing}")
            else:
                # Keep only required features in the correct order
                df_features = df[required_features]

                st.write("Uploaded Data Preview:")
                st.dataframe(df_features.head())

                # Load model and scaler
                model = joblib.load("models/credit_model.pkl")
                scaler = joblib.load("models/scaler.pkl")

                # Scale features
                df_scaled = scaler.transform(df_features)

                # Predict
                predictions = model.predict(df_scaled)

                df['Prediction'] = predictions

                st.write("Predictions:")
                st.dataframe(df[['Prediction']])

                # Save predictions CSV for download
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
