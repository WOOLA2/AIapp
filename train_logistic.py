import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load processed data
df = pd.read_csv("processed_data.csv")

# Prepare data
X = df.drop(['Target_encoded'], axis=1)
y = df['Target_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with increased iterations
lr_model = LogisticRegression(random_state=42, max_iter=2000)
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Comprehensive evaluation
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Dropout', 'Enrolled', 'Graduate']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(lr_model, 'logistic_model.pkl')
print("Model saved: logistic_model.pkl")
print("Logistic Regression training complete")