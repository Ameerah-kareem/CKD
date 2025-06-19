import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and prepare your dataset
df = pd.read_csv('kidney_disease.csv')
df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

# Preprocess
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

X = df[numeric_cols]
y = df['classification'].apply(lambda x: 1 if x.strip().lower() == 'ckd' else 0)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model correctly
with open('ckd_model.pkl', 'wb') as f:
    pickle.dump(model, f)
