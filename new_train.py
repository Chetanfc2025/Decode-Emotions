import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler

# ======================== Load Data ========================
LABEL_FILE_PATH = r"D:\Chetan\Msc 1\Final 2.0\label_list.txt"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isfile(LABEL_FILE_PATH):
    raise FileNotFoundError("❌ Error: label_list.txt file not found.")

print("✅ Info: label_list.txt found.")

# Initialize feature & label lists
features = {"baseline_angle": [], "top_margin": [], "letter_size": [], "line_spacing": [],
            "word_spacing": [], "pen_pressure": [], "slant_angle": []}
labels = []

with open(LABEL_FILE_PATH, "r") as labels_file:
    for line in labels_file:
        content = line.split()
        for i, key in enumerate(features.keys()):
            features[key].append(float(content[i]) + np.random.uniform(-0.5, 0.5))  # Adding small noise
        labels.append(int(content[7]))  # Assuming t1 (Emotional Stability) as the main target label

# ======================== Prepare Feature Set ========================
X = np.array([[features["baseline_angle"][i], features["slant_angle"][i], features["letter_size"][i], 
               features["pen_pressure"][i], features["top_margin"][i], features["line_spacing"][i], 
               features["word_spacing"][i]] for i in range(len(labels))])

y = np.array(labels)

# ======================== Data Scaling ========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================== Train Model ========================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=42)
for train_idx, test_idx in sss.split(X_scaled, y):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Train SVM Model
clf = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ======================== Model Metrics ========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
cv_scores = cross_val_score(clf, X_scaled, y, cv=3, scoring='accuracy')

try:
    overall_auc = roc_auc_score(y_test, y_pred)
except ValueError:
    overall_auc = "N/A (Only one class present in y_true)"

overall_conf_matrix = confusion_matrix(y_test, y_pred)

# Print Metrics
print("\n📊 **Overall Performance Metrics:**")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔄 Recall: {recall:.4f}")
print(f"📊 F1 Score: {f1:.4f}")
print(f"📈 CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"📈 AUC Score: {overall_auc}")
print(f"📌 Confusion Matrix:\n{overall_conf_matrix}")

# ======================== Save Model & Scaler ========================
model_path = os.path.join(MODEL_DIR, "model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

joblib.dump(clf, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n💾 Model saved as: {model_path}")
print(f"💾 Scaler saved as: {scaler_path}")

# ======================== Plot Confusion Matrix ========================
plt.figure(figsize=(6, 5))
sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Overall Confusion Matrix")
plt.show()

print("\n✅ Training Complete. Model and scaler saved.")
