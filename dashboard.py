import streamlit as st
import pandas as pd
import joblib
import numpy as np
from preprocessing_pipeline import StudentDataPreprocessor

# Page config
st.set_page_config(page_title="Luct AI", layout="wide")

# Custom CSS for purple and pink Prada-inspired styling
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #8B008B 0%, #FF69B4 50%, #DA70D6 100%);
    }
    .luxury-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(139, 0, 139, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .luxury-metric {
        background: linear-gradient(135deg, #8B008B 0%, #DA70D6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #8B008B 0%, #FF69B4 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 0, 139, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF69B4 0%, #8B008B 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 0, 139, 0.4);
    }
    .stDownloadButton>button {
        background: linear-gradient(135deg, #9370DB 0%, #BA55D3 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(147, 112, 219, 0.3);
    }
    .stFileUploader > div > div > button {
        background: linear-gradient(135deg, #C71585 0%, #DB7093 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(199, 21, 133, 0.3);
    }
    .stRadio > div {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(139, 0, 139, 0.2);
    }
    .stExpander {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(139, 0, 139, 0.2);
        border-radius: 12px;
    }
    .prediction-high {
        background: linear-gradient(135deg, #FFB6C1 0%, #FF69B4 100%);
        color: #8B0000;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 5px solid #8B0000;
        box-shadow: 0 4px 15px rgba(255, 105, 180, 0.2);
    }
    .prediction-medium {
        background: linear-gradient(135deg, #D8BFD8 0%, #DA70D6 100%);
        color: #4B0082;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 5px solid #4B0082;
        box-shadow: 0 4px 15px rgba(218, 112, 214, 0.2);
    }
    .prediction-low {
        background: linear-gradient(135deg, #E6E6FA 0%, #9370DB 100%);
        color: #2E8B57;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 5px solid #2E8B57;
        box-shadow: 0 4px 15px rgba(147, 112, 219, 0.2);
    }
    .main-title {
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-family: 'Arial', sans-serif;
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 3rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        model = joblib.load('tuned_student_model_compressed.pkl')
        expected_features = joblib.load('expected_features_compressed.pkl')
        preprocessor = StudentDataPreprocessor()
        return model, expected_features, preprocessor, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, False


# Header
st.markdown("<h1 class='main-title'>Luct AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Academic Performance Analytics Platform</p>", unsafe_allow_html=True)

model, expected_features, preprocessor, models_loaded = load_models()

if not models_loaded:
    st.error("Required models or preprocessing files not found. Please run data processing first.")
    st.stop()

def analyze_data_quality(df):
    """Analyze data quality including missing values and outliers"""
    quality_report = {
        'missing_values': df.isnull().sum().sum(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'total_records': len(df),
        'total_features': len(df.columns)
    }
    return quality_report

def process_and_predict(uploaded_data):
    """Complete pipeline: clean -> process -> align -> predict"""
    try:
        with st.spinner("Cleaning data..."):
            cleaned_data = preprocessor.clean_data(uploaded_data)
        
        with st.spinner("Processing data..."):
            processed_data = preprocessor.process_data(cleaned_data)
        
        with st.spinner("Aligning with model features..."):
            final_data = preprocessor.align_with_training_features(processed_data, expected_features)
        
        with st.spinner("Making predictions..."):
            predictions = model.predict(final_data)
            probabilities = model.predict_proba(final_data)
        
        return predictions, probabilities, final_data, cleaned_data, None
        
    except Exception as e:
        return None, None, None, None, str(e)

# Main content
st.markdown('<div class="luxury-card">', unsafe_allow_html=True)

option = st.radio("Choose prediction method:", 
                 ["Upload CSV & Select Rows", "Manual Data Entry"], horizontal=True)

st.markdown("---")

if option == "Upload CSV & Select Rows":
    st.subheader("Upload Student Data")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            student_data = pd.read_csv(uploaded_file)
            
            st.success(f"Loaded {len(student_data)} student records with {len(student_data.columns)} features")
            
            # Data quality analysis
            quality_report = analyze_data_quality(student_data)
            
            with st.expander("Raw Data Preview"):
                st.dataframe(student_data.head())
            
            # Data quality info
            st.subheader("Data Quality Check")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="luxury-metric"><h3>{quality_report["missing_values"]}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="luxury-metric"><h3>{quality_report["total_records"]}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="luxury-metric"><h3>{quality_report["total_features"]}</h3><p>Features</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="luxury-metric"><h3>{len(expected_features)}</h3><p>Expected Features</p></div>', unsafe_allow_html=True)
            
            # Show missing values by column if any
            if quality_report['missing_values'] > 0:
                with st.expander("Missing Values Details"):
                    missing_cols = {k: v for k, v in quality_report['missing_by_column'].items() if v > 0}
                    for col, count in missing_cols.items():
                        st.write(f"**{col}:** {count} missing values")
            
            selected_rows = st.multiselect(
                "Choose specific student rows to predict", 
                options=list(range(len(student_data))),
                format_func=lambda x: f"Row {x} - Student {x+1}"
            )
            
            if selected_rows:
                st.subheader("Selected Students Data")
                selected_data = student_data.iloc[selected_rows]
                st.dataframe(selected_data)
                
                # Show data quality for selected rows
                selected_quality = analyze_data_quality(selected_data)
                if selected_quality['missing_values'] > 0:
                    st.warning(f"Selected rows contain {selected_quality['missing_values']} missing values")
            
            if st.button("Predict Selected Students") and selected_rows:
                selected_students = student_data.iloc[selected_rows]
                
                predictions, probabilities, processed_data, cleaned_data, error = process_and_predict(selected_students)
                
                if error:
                    st.error(f"Prediction failed: {error}")
                else:
                    # Display results
                    labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
                    results = []
                    
                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        results.append({
                            'Row': selected_rows[i],
                            'Student ID': f"Student {selected_rows[i]+1}",
                            'Prediction': labels[pred],
                            'Dropout Probability': f"{prob[0]:.1%}",
                            'Enrolled Probability': f"{prob[1]:.1%}",
                            'Graduate Probability': f"{prob[2]:.1%}",
                            'Risk Level': 'HIGH' if prob[0] > 0.7 else 'MEDIUM' if prob[0] > 0.3 else 'LOW'
                        })
                    
                    st.header("Prediction Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Display prediction cards
                    for i, result in enumerate(results):
                        risk_class = "prediction-high" if result['Risk Level'] == 'HIGH' else "prediction-medium" if result['Risk Level'] == 'MEDIUM' else "prediction-low"
                        st.markdown(f'''
                        <div class="{risk_class}">
                            <h4 style="margin: 0 0 8px 0;">{result["Student ID"]}</h4>
                            <h3 style="margin: 0 0 8px 0;">{result["Prediction"]}</h3>
                            <p style="margin: 4px 0;"><strong>Risk Level:</strong> {result["Risk Level"]}</p>
                            <p style="margin: 4px 0;"><strong>Dropout Probability:</strong> {result["Dropout Probability"]}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Predictions as CSV",
                        csv,
                        "student_predictions.csv",
                        "text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error loading file: {e}")

else:  # Manual Data Entry
    st.subheader("Manual Student Data Entry")
    
    # Create sample input data with NaN for missing values
    sample_data = {}
    for feature in expected_features:
        sample_data[feature] = ""
    
    # Create editable dataframe with text columns
    editable_df = pd.DataFrame([sample_data])
    
    # Display data editor with text input for all columns
    st.write("Enter student data below (leave cells empty for missing values):")
    
    edited_df = st.data_editor(
        editable_df,
        height=300,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            col: st.column_config.TextColumn(
                col,
                help=f"Enter value for {col} or leave empty for missing",
                default=""
            ) for col in expected_features
        }
    )
    
    # Process the text input and convert to appropriate types
    def process_manual_input(df):
        processed_df = df.copy()
        
        for col in processed_df.columns:
            # Convert empty strings and text NaN to actual NaN
            processed_df[col] = processed_df[col].replace(['', 'nan', 'NaN', 'null', 'NULL', 'N/A', 'n/a'], np.nan)
            
            # Try to convert to numeric where possible, leave as string if not possible
            processed_df[col] = pd.to_numeric(processed_df[col], errors='ignore')
            
        return processed_df
    
    processed_manual_df = process_manual_input(edited_df)
    
    # Show data quality for manual entries
    if len(processed_manual_df) > 0:
        manual_quality = analyze_data_quality(processed_manual_df)
        
        st.subheader("Data Quality Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Students Entered", len(processed_manual_df))
        with col2:
            st.metric("Missing Values", manual_quality['missing_values'])
        
        if manual_quality['missing_values'] > 0:
            with st.expander("Missing Values in Your Data"):
                missing_cols = {k: v for k, v in manual_quality['missing_by_column'].items() if v > 0}
                for col, count in missing_cols.items():
                    st.write(f"**{col}:** {count} missing value(s)")
        
        # Show preview of processed data
        with st.expander("Preview Your Entered Data"):
            st.write("This is how your data will be processed:")
            st.dataframe(processed_manual_df)
    
    if st.button("Predict Manual Entries", type="primary", use_container_width=True):
        if len(processed_manual_df) == 0:
            st.warning("Please enter at least one student record")
        else:
            # Check if all rows are empty
            if processed_manual_df.isna().all(axis=1).any():
                st.error("Please enter valid data for at least some features in each student row")
            else:
                predictions, probabilities, processed_data, cleaned_data, error = process_and_predict(processed_manual_df)
                
                if error:
                    st.error(f"Prediction failed: {error}")
                else:
                    labels = {0: 'DROPOUT', 1: 'ENROLLED', 2: 'GRADUATE'}
                    results = []
                    
                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        results.append({
                            "Student": f"Row {i+1}",
                            "Prediction": labels[pred],
                            "Dropout Probability": f"{prob[0]:.1%}",
                            "Enrolled Probability": f"{prob[1]:.1%}",
                            "Graduate Probability": f"{prob[2]:.1%}",
                            "Risk Level": "HIGH" if prob[0] > 0.7 else "MEDIUM" if prob[0] > 0.3 else "LOW"
                        })
                    
                    results_df = pd.DataFrame(results)
                    st.success(f"Predicted {len(results_df)} student(s) successfully!")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Display prediction cards
                    for result in results:
                        risk_class = "prediction-high" if result['Risk Level'] == 'HIGH' else "prediction-medium" if result['Risk Level'] == 'MEDIUM' else "prediction-low"
                        st.markdown(f'''
                        <div class="{risk_class}">
                            <h4 style="margin: 0 0 8px 0;">{result["Student"]}</h4>
                            <h3 style="margin: 0 0 8px 0;">{result["Prediction"]}</h3>
                            <p style="margin: 4px 0;"><strong>Risk Level:</strong> {result["Risk Level"]}</p>
                            <p style="margin: 4px 0;"><strong>Dropout Probability:</strong> {result["Dropout Probability"]}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "manual_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )

st.markdown('</div>', unsafe_allow_html=True)