import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("processed_data.csv")

# Prepare data
X = df.drop(['Target_encoded'], axis=1)
y = df['Target_encoded'] 

# Enhanced parameter grid
rf = RandomForestClassifier(random_state=109)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],       
    'max_features': ['sqrt', 'log2'],  
    'criterion': ['gini', 'entropy']
}


print("Tuning with expanded parameter grid")
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
 
# Save tuned model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'tuned_student_model.pkl')
print("Tuned model saved: tuned_student_model.pkl")
print("Model tuning complete")