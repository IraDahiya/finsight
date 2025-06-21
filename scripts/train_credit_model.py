import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load data from local CSV file in the current folder
df = pd.read_csv('credit_data.csv', skiprows=1)  # skiprows=1 if CSV has extra header row

# Drop unwanted columns if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Define feature columns expected by your model
feature_cols = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Target column
target_col = 'default payment next month'  # Adjust if your CSV has a different name

# Select features and target
X = df[feature_cols]
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (using Logistic Regression here, but you can change)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'models/credit_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Model training complete and saved!")
