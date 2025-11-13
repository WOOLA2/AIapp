print("=== STUDENT DROPOUT PREDICTION PIPELINE ===")

print("\n1. 📊 Data gathering and cleaning...")
import dataload

print("\n2. 🔧 Data preprocessing...")
import processing

print("\n3. 🤖 Model training...")
print("   Training Logistic Regression...")
import train_logistic

print("   Training Random Forest...")
import train_random

print("\n4. ⚙️ Model tuning...")
import model_tuning

print("\n5. 📁 Creating test data...")
import create_testing_data

print("\n🎉 PIPELINE COMPLETE!")
print("👉 Next: Run 'streamlit run dashboard.py' for the prediction dashboard")
print("👉 Don't forget to write your 10+ page report!")