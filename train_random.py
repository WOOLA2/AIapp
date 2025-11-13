import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    """Train Random Forest model and return it"""
    df = pd.read_csv("processed_data.csv")
    X = df.drop(['Target_encoded'], axis=1)
    y = df['Target_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=109)
    rf_model.fit(X_train, y_train)
    
    return rf_model, X_test, y_test

# For standalone execution
if __name__ == "__main__":
    model, X_test, y_test = train_model()
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate']))
    
    joblib.dump(model, 'student_dropout_model.pkl')
    print("Model saved!")