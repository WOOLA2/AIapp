import pandas as pd
import numpy as np

print("=== CREATING TEST DATA ===")

# Load original data
df_original = pd.read_csv(r"C:\Users\Molefi\Desktop\ProjectAI\studata.csv", sep=';')

print(f"Original data shape: {df_original.shape}")

# Create test data without target
df_test = df_original.drop('Target', axis=1)

print(f"Test data shape: {df_test.shape}")
print(f"Number of features: {len(df_test.columns)}")

# Save test data
df_test.to_csv("no_target.csv", index=False)

print("\n✅ Created: no_target.csv")
print("✅ Ready for dashboard testing!")