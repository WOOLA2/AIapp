import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\Molefi\Desktop\ProjectAI\cleaned_data.csv")  # Default comma separator

print("=== TASK 2: DATA PREPROCESSING ===")

# Fix: Use only features that actually exist
df['Success_rate_1st'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)'].replace(0, 1)
df['Success_rate_2nd'] = df['Curricular units 2nd sem (approved)'] / df['Curricular units 2nd sem (enrolled)'].replace(0, 1)
df['Grade_improvement'] = df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']

# Fix: Only remove truly useless columns (keep economic indicators)
columns_to_remove = ['Curricular units 1st sem (without evaluations)', 
                    'Curricular units 2nd sem (without evaluations)']
df_clean = df.drop(columns=columns_to_remove)

print("✅ Created 3 meaningful features from existing data")
print("✅ Kept important economic indicators")

# PROPERLY encode categorical target
le = LabelEncoder()
df_clean['Target_encoded'] = le.fit_transform(df_clean['Target'])

# Save WITHOUT the original Target column to avoid confusion
df_clean = df_clean.drop('Target', axis=1)
df_clean.to_csv("processed_student_data.csv", index=False)
print("✅ Task 2 complete: Proper feature engineering and encoding")