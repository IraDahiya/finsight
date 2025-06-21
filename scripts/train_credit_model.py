import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load CSV with skiprows=1 to ignore extra header row
df = pd.read_csv("data/credit_data.csv", skiprows=1)

# Drop ID column if present
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Check your target column name, e.g. "default payment next month"
target_col = "default payment next month"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in data")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save scaler and model
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(model, "models/credit_model.pkl")

print("Model training complete and saved!")
