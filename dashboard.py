import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="wide")

# Load model and data
@st.cache_resource
def load_model():
    try:
        model = joblib.load('student_dropout_model.pkl')
        return model, True
    except:
        return None, False

# Header
st.title("🎓 Student Dropout Prediction Dashboard")
st.markdown("Predict student academic outcomes using machine learning")

model, model_loaded = load_model()

if not model_loaded:
    st.error("❌ Model not found. Please train the model first.")
    st.stop()

st.success(f"✅ Model loaded successfully! (Expected {model.n_features_in_} features)")

# Load original feature names
@st.cache_data
def get_original_features():
    try:
        df = pd.read_csv(r"C:\Users\Molefi\Desktop\ProjectAI\studata.csv", sep=';')
        return df.drop('Target', axis=1).columns.tolist()
    except:
        return None

original_features = get_original_features()

# Sidebar
st.sidebar.header("📊 Model Info")
st.sidebar.write(f"**Model Type:** Random Forest")
st.sidebar.write(f"**Input Features:** {model.n_features_in_}")
st.sidebar.write(f"**Classes:** Dropout, Enrolled, Graduate")

st.sidebar.header("🎯 Instructions")
st.sidebar.write("""
**Option 1 - CSV Upload:**
- Upload student data CSV
- Select multiple rows to predict

**Option 2 - Manual Entry:**
- Enter values for all features

**🎯 Risk Levels:**
- 🔴 HIGH: >70% dropout probability  
- 🟡 MEDIUM: 30-70% dropout probability
- 🟢 LOW: <30% dropout probability
""")

# Main content
option = st.radio("Choose prediction method:", 
                 ["📁 Upload CSV & Select Rows", "✍️ Manual Data Entry"], horizontal=True)

if option == "📁 Upload CSV & Select Rows":
    st.subheader("Upload Student Data and Select Specific Students")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        student_data = pd.read_csv(uploaded_file)
        
        if len(student_data.columns) != model.n_features_in_:
            st.error(f"❌ Wrong number of features! Expected {model.n_features_in_}, got {len(student_data.columns)}")
        else:
            st.success(f"✅ Loaded {len(student_data)} student records")
            
            with st.expander("📋 Full Data Preview"):
                st.dataframe(student_data)
            
            selected_rows = st.multiselect(
                "Choose specific student rows to predict", 
                options=list(range(len(student_data))),
                format_func=lambda x: f"Row {x} - Student {x+1}"
            )
            
            if selected_rows:
                st.subheader("📋 Selected Students Data")
                selected_data = student_data.iloc[selected_rows]
                st.dataframe(selected_data)
            
            if st.button("🚀 Predict Selected Students") and selected_rows:
                predictions = []
                
                for row_num in selected_rows:
                    student_row = student_data.iloc[row_num]
                    sample_data = np.array([student_row.values])
                    
                    try:
                        prediction = model.predict(sample_data)[0]
                        pred_proba = model.predict_proba(sample_data)[0]
                        
                        labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
                        
                        predictions.append({
                            'Row': row_num,
                            'Student ID': f"Student {row_num+1}",
                            'Prediction': labels[prediction],
                            'Dropout Probability': f"{pred_proba[0]:.1%}",
                            'Enrolled Probability': f"{pred_proba[1]:.1%}",
                            'Graduate Probability': f"{pred_proba[2]:.1%}",
                            'Risk Level': 'HIGH' if pred_proba[0] > 0.7 else 'MEDIUM' if pred_proba[0] > 0.3 else 'LOW'
                        })
                        
                    except Exception as e:
                        st.error(f"Error predicting row {row_num}: {e}")
                
                if predictions:
                    st.header("📊 Batch Prediction Results")
                    results_df = pd.DataFrame(predictions)
                    st.dataframe(results_df)
                    
                    st.subheader("🎯 Risk Level Distribution")
                    risk_counts = results_df['Risk Level'].value_counts()
                    st.bar_chart(risk_counts)
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions as CSV",
                        csv,
                        "student_predictions.csv",
                        "text/csv"
                    )
else:  # Manual Data Entry
    st.write("Enter one or more students manually for prediction")

    # Load feature names
    if original_features and len(original_features) == model.n_features_in_:
        feature_names = original_features
        st.success(f"✅ Using all {len(feature_names)} original feature names")
    else:
        feature_names = [f"Feature_{i}" for i in range(model.n_features_in_)]
        st.warning(f"Using generic names for {model.n_features_in_} features")

    # Feature guidance for better defaults
    feature_guides = {
        'Admission grade': (0, 200, 100),
        'Curricular units 1st sem (grade)': (0, 20, 10),
        'Curricular units 2nd sem (grade)': (0, 20, 10),
        'Age at enrollment': (16, 60, 20),
        'Unemployment rate': (0, 30, 5),
        'Inflation rate': (-10, 50, 2),
        'Previous qualification (grade)': (0, 200, 100),
        'Gender': (0, 1, 1),
        'Debtor': (0, 1, 0),
        'Scholarship holder': (0, 1, 0),
        'International': (0, 1, 0)
    }

    # Create one default student row
    default_row = {}
    for feature in feature_names:
        if feature in feature_guides:
            _, _, default_val = feature_guides[feature]
            default_row[feature] = default_val
        else:
            default_row[feature] = 0

    st.subheader("🧾 Enter One or More Students (Editable Table)")

    # Editable table with add/delete row functionality
    editable_df = pd.DataFrame([default_row])

    edited_df = st.data_editor(
        editable_df,
        height=280,        # ~5 visible rows (scroll vertically)
        width=950,         # ~5 visible columns (scroll horizontally)
        num_rows="dynamic" # Enables ➕ Add and 🗑️ Delete row buttons
    )

    # Download current table as CSV
    st.download_button(
        "📥 Download Entered Data as CSV",
        edited_df.to_csv(index=False),
        "manual_student_data.csv",
        "text/csv",
        use_container_width=True
    )

    # Predict outcomes for all entered rows
    if st.button("🔮 Predict Outcomes", type="primary", use_container_width=True):
        try:
            # Convert table data to numpy array
            sample_data = edited_df.values

            # Run model predictions
            predictions = model.predict(sample_data)
            probas = model.predict_proba(sample_data)

            labels = {0: '🚨 DROPOUT', 1: '📚 ENROLLED', 2: '🎓 GRADUATE'}
            results = []

            for i, pred in enumerate(predictions):
                results.append({
                    "Student": f"Row {i+1}",
                    "Prediction": labels[pred],
                    "Dropout Probability": f"{probas[i][0]:.1%}",
                    "Enrolled Probability": f"{probas[i][1]:.1%}",
                    "Graduate Probability": f"{probas[i][2]:.1%}",
                    "Risk Level": (
                        "🔴 HIGH" if probas[i][0] > 0.7
                        else "🟡 MEDIUM" if probas[i][0] > 0.3
                        else "🟢 LOW"
                    )
                })

            results_df = pd.DataFrame(results)

            # Show prediction results
            st.success(f"✅ Predicted {len(results_df)} student(s) successfully!")

            st.subheader("📊 Prediction Results")
            st.dataframe(results_df, height=250, use_container_width=True)

            # Risk distribution chart
            st.subheader("🎯 Risk Level Distribution")
            risk_counts = results_df["Risk Level"].value_counts()
            st.bar_chart(risk_counts)

            # Allow downloading predictions
            st.download_button(
                "📊 Download Predictions as CSV",
                results_df.to_csv(index=False),
                "student_predictions.csv",
                "text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# Footer
st.markdown("---")
st.markdown("**BIAI 3110 • Artificial Intelligence • Group Assignment**")