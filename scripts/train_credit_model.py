import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load preprocessed data
df = pd.read_csv('data/credit_data_processed.csv')

# Check columns
print("Columns in dataset:", df.columns.tolist())

# Features and target
X = df.drop(columns=['Y'])
y = df['Y']

# Convert target to int if it's float with discrete values
print("Target dtype before conversion:", y.dtype)
y = y.astype(int)
print("Target dtype after conversion:", y.dtype)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and the scaler
joblib.dump(model, 'models/credit_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Model and scaler saved successfully.")

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))
