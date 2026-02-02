import os
import shutil
import matplotlib.pyplot as plt
from utils import load_data, generate_synthetic_data
from model_builder import build_baseline_model, build_multiscale_model

DATA_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Regenerate Data for new classes
# We force regeneration to ensure we have the RGB/Multi-class data
print("Regenerating dataset with new classes (Clean, Rust, Broken, Dusty)...")
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR) # Clear old binary data
generate_synthetic_data(DATA_DIR, num_samples=250)

# 2. Load Data
print("Loading data...")
X_train, X_test, y_train, y_test, class_names = load_data(DATA_DIR)
print(f"Classes: {class_names}")
print(f"Train/Test split: {len(X_train)}/{len(X_test)}")

# 3. Train Baseline
print("\n--- Training Baseline Model ---")
baseline = build_baseline_model()
h1 = baseline.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test), batch_size=32)
baseline.save(os.path.join(MODEL_DIR, "baseline.h5"))

# 4. Train Multi-Scale
print("\n--- Training Multi-Scale Model ---")
multiscale = build_multiscale_model()
h2 = multiscale.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test), batch_size=32)
multiscale.save(os.path.join(MODEL_DIR, "multiscale.h5"))

# 5. Evaluate & Compare
print("\n--- Evaluation ---")
loss1, acc1 = baseline.evaluate(X_test, y_test, verbose=0)
loss2, acc2 = multiscale.evaluate(X_test, y_test, verbose=0)

print(f"Baseline Accuracy: {acc1:.4f}")
print(f"Multi-Scale Accuracy: {acc2:.4f}")

# Optional: Plot history
plt.figure()
plt.plot(h1.history['val_accuracy'], label='Baseline Val Acc')
plt.plot(h2.history['val_accuracy'], label='Multi-Scale Val Acc')
plt.legend()
plt.title("Model Comparison (Multi-Class)")
plt.savefig("comparison_plot_multiclass.png")
print("Comparison plot saved.")
