import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed data
df = pd.read_csv("processed_student_data.csv")

# Prepare data - NOW USE THE ENCODED TARGET
X = df.drop(['Target_encoded'], axis=1)
y = df['Target_encoded']  # Use the already encoded target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== TASK 4: RANDOM FOREST TRAINING ===")
print("🤔 Model Choice: Random Forest (handles non-linear data, robust to outliers)")

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Comprehensive evaluation
print(f"📊 Accuracy: {accuracy:.3f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Dropout', 'Enrolled', 'Graduate']))

print("🎯 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model (for dashboard)
joblib.dump(rf_model, 'student_dropout_model.pkl')
print("💾 Model saved: student_dropout_model.pkl")
print("✅ Random Forest training complete")