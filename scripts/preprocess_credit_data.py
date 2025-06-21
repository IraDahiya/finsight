import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('data/credit_data.csv')

if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

le = LabelEncoder()
for col in ['SEX', 'EDUCATION', 'MARRIAGE']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Try converting all columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

numeric_features = df.select_dtypes(include=['number']).columns
print("Numeric features after conversion:", list(numeric_features))

# Fill NaNs in numeric columns (to avoid errors during scaling)
df[numeric_features] = df[numeric_features].fillna(0)

# Scale numeric features
df[numeric_features] = StandardScaler().fit_transform(df[numeric_features])

df.to_csv('data/credit_data_processed.csv', index=False)
print("✅ Preprocessing complete. Saved as data/credit_data_processed.csv")
