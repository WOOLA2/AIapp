print("=== STUDENT DROPOUT PREDICTION PIPELINE ===\n")

print("1. 📊 Data gathering and cleaning...")
exec(open("task1_cleaning.py").read())

print("\n2. 🔧 Data preprocessing...") 
exec(open("task2_preprocessing.py").read())

print("\n3. 🤖 Model training...")
exec(open("task4_training.py").read())

print("\n4. ⚙️ Model tuning...")
exec(open("task5_tuning.py").read())

print("\n5. 📁 Creating test data...")
exec(open("create_test_data.py").read())

print("\n🎉 PIPELINE COMPLETE!")
print("👉 Next: Run 'streamlit run dashboard.py' for the prediction dashboard")