import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import yfinance as yf

st.title("FinSight")

option = st.selectbox("Select Module", ["Credit Risk Prediction", "Stock Price Forecast"])

if option == "Credit Risk Prediction":
    st.header("Upload Credit Data for Default Prediction")
    file = st.file_uploader("Upload CSV file", type="csv")
    
    if file:
        try:
            df = pd.read_csv(file, skiprows=1)
            # Drop unwanted columns to match training features
            for col in ['Unnamed: 0', 'ID']:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Load saved model, scaler, and feature names
            model = joblib.load("models/credit_model.pkl")
            scaler = joblib.load("models/scaler.pkl")
            feature_names = joblib.load("models/feature_names.pkl")

            # Check if features match
            if list(df.columns) != feature_names:
                st.error(f"Feature mismatch! Please upload a file with columns:\n{feature_names}")
            else:
                st.write("Uploaded Data Preview:")
                st.dataframe(df.head())

                # Scale features
                df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

                # Predict
                predictions = model.predict(df_scaled)
                df['Prediction'] = predictions

                st.write("Predictions:")
                st.dataframe(df[['Prediction']])

                # Save and provide download button
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
            data = data.dropna()
            data['day'] = np.arange(len(data))
            X = data['day'].values.reshape(-1, 1)
            y = data['Close'].values
            
            model = LogisticRegression()
            model.fit(X, y)
            forecast = model.predict(X)
            st.line_chart(pd.DataFrame({'Actual': y, 'Forecast': forecast}))
            st.success("Forecast plotted above.")
