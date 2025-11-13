import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv(r"C:\Users\Molefi\Desktop\limkos\cleaned_sample.csv")

# Your existing processing code...
df['Success_rate_1st'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)'].replace(0, 1)
df['Success_rate_2nd'] = df['Curricular units 2nd sem (approved)'] / df['Curricular units 2nd sem (enrolled)'].replace(0, 1)
df['Grade_improvement'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']

columns_to_remove = ['Curricular units 1st sem (without evaluations)', 
                    'Curricular units 2nd sem (without evaluations)']
df_clean = df.drop(columns=columns_to_remove)

le = LabelEncoder()
df_clean['Target_encoded'] = le.fit_transform(df_clean['Target'])
df_clean = df_clean.drop('Target', axis=1)

# Save processed data
df_clean.to_csv("processed_data.csv", index=False)

# Save expected features for dashboard
expected_features = df_clean.drop('Target_encoded', axis=1).columns.tolist()
joblib.dump(expected_features, 'expected_features.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Task 2 complete: Proper feature engineering and encoding")
print(f"Expected features saved: {len(expected_features)} features")