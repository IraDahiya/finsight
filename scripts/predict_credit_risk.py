import pandas as pd
import joblib
import sys
import os

# Load new data
input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/new_customers.csv'
new_data = pd.read_csv(input_file)

# ✅ Add 'Unnamed: 0' if missing — scaler needs it
if 'Unnamed: 0' not in new_data.columns:
    new_data.insert(0, 'Unnamed: 0', range(len(new_data)))

# Load scaler and model
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/credit_model.pkl')

# Scale data
new_data_scaled = pd.DataFrame(
    scaler.transform(new_data),
    columns=new_data.columns
)

# Predict
predictions = model.predict(new_data_scaled)

# Save predictions
output = new_data.copy()
output['Prediction'] = predictions
output.to_csv('outputs/predictions.csv', index=False)

print("✅ Predictions saved to outputs/predictions.csv")
