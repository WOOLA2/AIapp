# fix_model.py
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load your original data with 36 features
df = pd.read_csv(r"C:\Users\Molefi\Desktop\ProjectAI\studata.csv", sep=';')

print(f"Original data shape: {df.shape}")  # Should be (4424, 37) including Target

# Handle missing values and outliers
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Prepare features (36 features) and target
X = df.drop('Target', axis=1)  # This should have 36 columns
y = df['Target'].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

print(f"Features shape: {X.shape}")  # Should be (4424, 36)
print(f"Feature names: {list(X.columns)}")

# Retrain model with correct 36 features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the corrected model
joblib.dump(model, 'student_dropout_model.pkl')
print(f"✅ Model retrained with {X.shape[1]} features")
print("✅ Model saved with correct feature count")