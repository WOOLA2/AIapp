import joblib

print("🔄 Loading original model...")
model = joblib.load("tuned_student_model.pkl")

print("💾 Compressing model...")
joblib.dump(model, "tuned_student_model_compressed.pkl", compress=("lzma", 3))

print("📁 Saving compressed expected_features...")
expected_features = joblib.load("expected_features.pkl")
joblib.dump(expected_features, "expected_features_compressed.pkl", compress=("lzma", 3))

print("✅ Compression complete!")
print("Created files: tuned_student_model_compressed.pkl, expected_features_compressed.pkl")
