import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data, skipping extra header row if present
df = pd.read_csv("data/credit_data.csv", skiprows=1)

# Drop unwanted columns if they exist
for col in ['Unnamed: 0', 'ID']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Separate target column 'default payment next month' (adjust if named differently)
target_col = 'default payment next month'

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in data")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "models/credit_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save feature names for prediction phase
feature_names = list(X_train.columns)
joblib.dump(feature_names, "models/feature_names.pkl")

print("Model training complete and saved!")
