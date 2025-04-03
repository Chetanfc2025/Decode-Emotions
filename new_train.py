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
labels = {"t1": [], "t2": [], "t3": [], "t4": [], "t5": [], "t6": [], "t7": [], "t8": []}
page_ids = []

with open(LABEL_FILE_PATH, "r") as labels_file:
    for line in labels_file:
        content = line.split()
        for i, key in enumerate(features.keys()):
            features[key].append(float(content[i]) + np.random.uniform(-0.5, 0.5))  # Adding small noise
        for j, key in enumerate(labels.keys(), start=7):
            labels[key].append(int(content[j]))
        page_ids.append(content[15])

# ======================== Feature Sets ========================
feature_sets = {
    "Emotional Stability": (features["baseline_angle"], features["slant_angle"], labels["t1"]),
    "Mental Energy/Will Power": (features["letter_size"], features["pen_pressure"], labels["t2"]),
    "Modesty": (features["letter_size"], features["top_margin"], labels["t3"]),
    "Personal Harmony/Flexibility": (features["line_spacing"], features["word_spacing"], labels["t4"]),
    "Lack of Discipline": (features["slant_angle"], features["top_margin"], labels["t5"]),
    "Poor Concentration": (features["letter_size"], features["line_spacing"], labels["t6"]),
    "Non-communicativeness": (features["letter_size"], features["word_spacing"], labels["t7"]),
    "Social Isolation": (features["line_spacing"], features["word_spacing"], labels["t8"]),
}

# ======================== Train Classifiers ========================
trained_classifiers = {}
all_y_true, all_y_pred = [], []

for trait, (X1, X2, y) in feature_sets.items():
    X = np.array([[a, b] for a, b in zip(X1, X2)])
    y = np.array(y)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=42)  # Using larger test size
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    # Using a simpler model with higher regularization
    clf = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    cv_scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')  # Reduced cross-validation

    # Print Metrics
    print(f"\n🔍 {trait} Classifier:")
    print(f"  ✅ Accuracy: {accuracy:.4f}")
    print(f"  🎯 Precision: {precision:.4f}")
    print(f"  🔄 Recall: {recall:.4f}")
    print(f"  📊 F1 Score: {f1:.4f}")
    print(f"  📈 CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Save trained model
    safe_trait_name = re.sub(r"[^\w\-_]", "_", trait.lower())
    model_path = os.path.join(MODEL_DIR, f"{safe_trait_name}_classifier.pkl")
    joblib.dump(clf, model_path)
    print(f"💾 Model saved: {model_path}")
    trained_classifiers[trait] = clf

# ======================== Overall Performance ========================
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
overall_precision = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=1)
overall_recall = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=1)
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=1)

try:
    overall_auc = roc_auc_score(all_y_true, all_y_pred)
except ValueError:
    overall_auc = "N/A (Only one class present in y_true)"

overall_conf_matrix = confusion_matrix(all_y_true, all_y_pred)

print("\n📊 **Overall Performance Metrics:**")
print(f"✅ Accuracy: {overall_accuracy:.4f}")
print(f"🎯 Precision: {overall_precision:.4f}")
print(f"🔄 Recall: {overall_recall:.4f}")
print(f"📊 F1 Score: {overall_f1:.4f}")
print(f"📈 AUC Score: {overall_auc}")
print(f"📌 Confusion Matrix:\n{overall_conf_matrix}")

# ======================== Plot Confusion Matrix ========================
plt.figure(figsize=(6, 5))
sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Overall Confusion Matrix")
plt.show()

print("\n✅ Training Complete. Models saved in 'models/' directory.")
