import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('data/credit_data.csv')

# Drop 'Unnamed: 0' if it exists (usually an index column)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Separate features and target
X = df.drop(columns=['Y'])  # Make sure 'Y' is your target column name
y = df['Y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save scaler and model
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(model, 'models/credit_model.pkl')

print("Model and scaler saved successfully!")
