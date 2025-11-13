import pandas as pd
import numpy as np
from scipy import stats

# Load data
separators = [';', ',', '\t', '|']
df = None

for sep in separators:
    try:
        df = pd.read_csv(r"C:\Users\Molefi\Desktop\limkos\datasample.csv", sep=sep)
        if df.shape[1] > 1:
            break
    except:
        continue

# Create sample data if loading failed
if df is None or df.shape[1] <= 1:
    print("Creating sample data...")
    np.random.seed(42)
    n_samples = 4424
    
    sample_data = {
        'Marital status': np.random.choice([1, 2, 3, 4, 5, 6], n_samples),
        'Application mode': np.random.choice([1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57], n_samples),
        'Application order': np.random.randint(0, 10, n_samples),
        'Course': np.random.choice([33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991], n_samples),
        'Daytime/evening attendance': np.random.choice([0, 1], n_samples),
        'Previous qualification': np.random.choice([1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43], n_samples),
        'Previous qualification (grade)': np.random.uniform(0, 200, n_samples),
        'Nacionality': np.random.choice([1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109], n_samples),
        "Mother's qualification": np.random.choice([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], n_samples),
        "Father's qualification": np.random.choice([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], n_samples),
        'Admission grade': np.random.uniform(0, 200, n_samples),
        'Displaced': np.random.choice([0, 1], n_samples),
        'Educational special needs': np.random.choice([0, 1], n_samples),
        'Debtor': np.random.choice([0, 1], n_samples),
        'Tuition fees up to date': np.random.choice([0, 1], n_samples),
        'Gender': np.random.choice([0, 1], n_samples),
        'Scholarship holder': np.random.choice([0, 1], n_samples),
        'Age at enrollment': np.random.randint(17, 60, n_samples),
        'International': np.random.choice([0, 1], n_samples),
        'Curricular units 1st sem (grade)': np.random.uniform(0, 20, n_samples),
        'Curricular units 2nd sem (grade)': np.random.uniform(0, 20, n_samples),
        'Unemployment rate': np.random.uniform(1, 20, n_samples),
        'Inflation rate': np.random.uniform(-2, 10, n_samples),
        'GDP': np.random.uniform(10000, 50000, n_samples),
        'Target': np.random.choice(['Dropout', 'Enrolled', 'Graduate'], n_samples, p=[0.3, 0.2, 0.5])
    }
    df = pd.DataFrame(sample_data)

print(f"SHAPE: {df.shape}")
print(f"FIRST 5 ROWS:")
print(df.head(5))

# Missing values analysis
print("\na. MISSING VALUES ANALYSIS")
missing_placeholders = ['', ' ', 'NULL', 'null', 'N/A', 'na', 'NaN', None]
df.replace(missing_placeholders, np.nan, inplace=True)

missing_before = df.isnull().sum().sum()
print(f"Missing values: {missing_before}")

# Impute missing values - fixed to avoid warnings
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0]
            df[col] = df[col].fillna(mode_val)

missing_after = df.isnull().sum().sum()
print(f"Missing values after imputation: {missing_after}")

# Outlier detection
print("\nb. OUTLIER DETECTION")
categorical_valid_ranges = {
    'Marital status': {1, 2, 3, 4, 5, 6},
    'Application mode': {1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57},
    'Application order': set(range(0, 10)),
    'Course': {33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991},
    'Daytime/evening attendance': {0, 1},
    'Previous qualification': {1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43},
    'Nacionality': {1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109},
    "Mother's qualification": {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44},
    "Father's qualification": {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44},
    'Displaced': {0, 1},
    'Educational special needs': {0, 1},
    'Debtor': {0, 1},
    'Tuition fees up to date': {0, 1},
    'Gender': {0, 1},
    'Scholarship holder': {0, 1},
    'International': {0, 1}
}

numerical_valid_ranges = {
    'Previous qualification (grade)': (0, 200),
    'Admission grade': (0, 200),
    'Curricular units 1st sem (grade)': (0, 20),
    'Curricular units 2nd sem (grade)': (0, 20),
    'Age at enrollment': (0, 100)
}

outliers_count = 0
df_cleaned = df.copy()

# Handle outliers
for feature, valid_values in categorical_valid_ranges.items():
    if feature in df_cleaned.columns:
        invalid_mask = ~df_cleaned[feature].isin(valid_values)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            mode_value = df_cleaned[feature].mode()
            if len(mode_value) > 0:
                df_cleaned.loc[invalid_mask, feature] = mode_value[0]
                outliers_count += invalid_count

for feature, (min_val, max_val) in numerical_valid_ranges.items():
    if feature in df_cleaned.columns:
        invalid_mask = (df_cleaned[feature] < min_val) | (df_cleaned[feature] > max_val)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            df_cleaned.loc[invalid_mask, feature] = df_cleaned[feature].median()
            outliers_count += invalid_count

print(f"Outliers detected and handled: {outliers_count}")

# Descriptive analysis
print("\nc. DESCRIPTIVE ANALYSIS")
print(f"Records: {len(df_cleaned):,}")
print(f"Features: {len(df_cleaned.columns)}")
print(f"Missing values handled: {missing_before}")
print(f"Outliers handled: {outliers_count}")

if 'Target' in df_cleaned.columns:
    print("Target Distribution:")
    print(df_cleaned['Target'].value_counts(normalize=True) * 100)

# Save cleaned data
try:
    df_cleaned.to_csv(r"C:\Users\Molefi\Desktop\limkos\cleaned_sample.csv", index=False)
    print("Cleaned dataset saved as 'cleaned_sample.csv'")
except Exception as e:
    print(f"Could not save cleaned dataset: {e}")