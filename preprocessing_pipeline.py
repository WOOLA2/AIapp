import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

class StudentDataPreprocessor:
    def __init__(self):
        self.categorical_valid_ranges = {
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
        
        self.numerical_valid_ranges = {
            'Previous qualification (grade)': (0, 200),
            'Admission grade': (0, 200),
            'Curricular units 1st sem (grade)': (0, 20),
            'Curricular units 2nd sem (grade)': (0, 20),
            'Age at enrollment': (0, 100),
            'Unemployment rate': (0, 30),
            'Inflation rate': (-10, 50),
            'GDP': (0, 100000)
        }
    
    def clean_data(self, df):
        """Apply same cleaning as in data_cleaning.py"""
        df_cleaned = df.copy()
        
        # Handle missing values
        missing_placeholders = ['', ' ', 'NULL', 'null', 'N/A', 'na', 'NaN', None]
        df_cleaned.replace(missing_placeholders, np.nan, inplace=True)
        
        # Impute missing values
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['float64', 'int64']:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                else:
                    mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else df_cleaned[col].iloc[0]
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val)
        
        # Handle outliers
        for feature, valid_values in self.categorical_valid_ranges.items():
            if feature in df_cleaned.columns:
                invalid_mask = ~df_cleaned[feature].isin(valid_values)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    mode_value = df_cleaned[feature].mode()
                    if len(mode_value) > 0:
                        df_cleaned.loc[invalid_mask, feature] = mode_value[0]
        
        for feature, (min_val, max_val) in self.numerical_valid_ranges.items():
            if feature in df_cleaned.columns:
                invalid_mask = (df_cleaned[feature] < min_val) | (df_cleaned[feature] > max_val)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    df_cleaned.loc[invalid_mask, feature] = df_cleaned[feature].median()
        
        return df_cleaned
    
    def process_data(self, df_cleaned):
        """Apply same processing as in data_processing.py"""
        df_processed = df_cleaned.copy()
        
        # Create engineered features (with error handling)
        try:
            if all(col in df_processed.columns for col in ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (enrolled)']):
                df_processed['Success_rate_1st'] = df_processed['Curricular units 1st sem (approved)'] / df_processed['Curricular units 1st sem (enrolled)'].replace(0, 1)
            
            if all(col in df_processed.columns for col in ['Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (enrolled)']):
                df_processed['Success_rate_2nd'] = df_processed['Curricular units 2nd sem (approved)'] / df_processed['Curricular units 2nd sem (enrolled)'].replace(0, 1)
            
            if all(col in df_processed.columns for col in ['Curricular units 2nd sem (grade)', 'Curricular units 1st sem (grade)']):
                df_processed['Grade_improvement'] = df_processed['Curricular units 2nd sem (grade)'] - df_processed['Curricular units 1st sem (grade)']
        except:
            pass
        
        # Remove unnecessary columns if they exist
        columns_to_remove = ['Curricular units 1st sem (without evaluations)', 
                            'Curricular units 2nd sem (without evaluations)', 'Target']
        for col in columns_to_remove:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
        
        return df_processed
    
    def align_with_training_features(self, df_processed, expected_features):
        """Ensure the data has exactly the same features as training data"""
        aligned_data = pd.DataFrame()
        
        for feature in expected_features:
            if feature in df_processed.columns:
                aligned_data[feature] = df_processed[feature]
            else:
                # Fill missing features with sensible defaults
                if 'grade' in feature.lower():
                    aligned_data[feature] = 10.0
                elif 'rate' in feature.lower():
                    aligned_data[feature] = 0.5
                elif 'improvement' in feature.lower():
                    aligned_data[feature] = 0.0
                elif feature in ['Gender', 'Debtor', 'Scholarship holder', 'International']:
                    aligned_data[feature] = 0
                else:
                    aligned_data[feature] = 0
        
        return aligned_data

def save_preprocessor():
    """Save the preprocessor for use in dashboard"""
    preprocessor = StudentDataPreprocessor()
    joblib.dump(preprocessor, 'data_preprocessor.pkl')

if __name__ == "__main__":
    save_preprocessor()
    print("Preprocessor saved!")