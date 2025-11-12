# task4_improved.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("=== TASK 4: MODEL TRAINING ===")

# Load processed data
df = pd.read_csv("processed_student_data.csv")

print(f"📊 Data shape: {df.shape}")

# Prepare features and target
X = df.drop(['Target', 'Target_encoded'], axis=1)
y = df['Target_encoded']

print(f"Features: {X.shape[1]}, Target classes: {y.nunique()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📋 DATA SPLIT:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
print(f"Feature count: {X_train.shape[1]}")

# Train Random Forest model
print("\n🤖 TRAINING RANDOM FOREST MODEL...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 MODEL PERFORMANCE:")
print(f"Accuracy: {accuracy:.3f}")

print("\n📋 CLASSIFICATION REPORT:")
class_names = ['Dropout', 'Enrolled', 'Graduate']
print(classification_report(y_test, y_pred, target_names=class_names))

print("🎯 CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
print(feature_importance.head(10))

# Save model and metadata
joblib.dump(model, 'student_dropout_model.pkl')
print("\n💾 Saved: student_dropout_model.pkl")

# Save feature names for dashboard
model_metadata = {
    'feature_names': list(X.columns),
    'class_names': class_names,
    'n_features': X.shape[1]
}
joblib.dump(model_metadata, 'model_metadata.pkl')
print("💾 Saved: model_metadata.pkl")

print("✅ TASK 4 COMPLETED SUCCESSFULLY!")