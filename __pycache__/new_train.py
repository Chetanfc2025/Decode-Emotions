import os
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import extract
import categorize

# Initialize feature and label lists
X_baseline_angle, X_top_margin, X_letter_size = [], [], []
X_line_spacing, X_word_spacing, X_pen_pressure, X_slant_angle = [], [], [], []
y_t1, y_t2, y_t3, y_t4, y_t5, y_t6, y_t7, y_t8 = [], [], [], [], [], [], [], []
page_ids = []

LABEL_FILE_PATH = r"D:\Chetan\Msc 1\FINAL\label_list_N.txt"

if os.path.isfile(LABEL_FILE_PATH):
    print("✅ Info: label_list.txt found.")

    with open(LABEL_FILE_PATH, "r") as labels:
        for line in labels:
            content = line.split()

            # Extract handwriting features
            X_baseline_angle.append(float(content[0]))
            X_top_margin.append(float(content[1]))
            X_letter_size.append(float(content[2]))
            X_line_spacing.append(float(content[3]))
            X_word_spacing.append(float(content[4]))
            X_pen_pressure.append(float(content[5]))
            X_slant_angle.append(float(content[6]))

            # Extract personality traits labels
            y_t1.append(int(content[7]))
            y_t2.append(int(content[8]))
            y_t3.append(int(content[9]))
            y_t4.append(int(content[10]))
            y_t5.append(int(content[11]))
            y_t6.append(int(content[12]))
            y_t7.append(int(content[13]))
            y_t8.append(int(content[14]))

            page_ids.append(content[15])

    # Feature sets for each personality trait
    feature_sets = {
        "Emotional Stability": (X_baseline_angle, X_slant_angle, y_t1),
        "Mental Energy/Will Power": (X_letter_size, X_pen_pressure, y_t2),
        "Modesty": (X_letter_size, X_top_margin, y_t3),
        "Personal Harmony/Flexibility": (X_line_spacing, X_word_spacing, y_t4),
        "Lack of Discipline": (X_slant_angle, X_top_margin, y_t5),
        "Poor Concentration": (X_letter_size, X_line_spacing, y_t6),
        "Non-communicativeness": (X_letter_size, X_word_spacing, y_t7),
        "Social Isolation": (X_line_spacing, X_word_spacing, y_t8),
    }

    trained_classifiers = {}

    # Train classifiers
    for trait, (X1, X2, y) in feature_sets.items():
        X = [[a, b] for a, b in zip(X1, X2)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = SVC(kernel='rbf', random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

        # Cross-validation
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        # Print results
        print(f"\n🔍 {trait} Classifier:")
        print(f"  ✅ Accuracy: {accuracy:.4f}")
        print(f"  🎯 Precision: {precision:.4f}")
        print(f"  🔄 Recall: {recall:.4f}")
        print(f"  📊 F1 Score: {f1:.4f}")
        print(f"  📈 Cross-validation scores: {cv_scores}")
        print(f"  📌 Mean CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Store trained classifier
        trained_classifiers[trait] = clf

    # ======================== Prediction Loop ========================
    while True:
        file_name = input("\n📁 Enter file name to predict or 'z' to exit: ")
        if file_name.lower() == 'z':
            break

        raw_features = extract.start(file_name)

        # Extract and categorize handwriting features
        feature_values = {
            "Baseline Angle": categorize.determine_baseline_angle(raw_features[0]),
            "Top Margin": categorize.determine_top_margin(raw_features[1]),
            "Letter Size": categorize.determine_letter_size(raw_features[2]),
            "Line Spacing": categorize.determine_line_spacing(raw_features[3]),
            "Word Spacing": categorize.determine_word_spacing(raw_features[4]),
            "Pen Pressure": categorize.determine_pen_pressure(raw_features[5]),
            "Slant Angle": categorize.determine_slant_angle(raw_features[6]),
        }

        for feature, (category, comment) in feature_values.items():
            print(f"🖊️ {feature}: {comment}")

        # Predict traits
        print("\n🔮 **Predicted Traits:**")
        predictions = {
            "Emotional Stability": trained_classifiers["Emotional Stability"].predict([[feature_values["Baseline Angle"][0], feature_values["Slant Angle"][0]]])[0],
            "Mental Energy/Will Power": trained_classifiers["Mental Energy/Will Power"].predict([[feature_values["Letter Size"][0], feature_values["Pen Pressure"][0]]])[0],
            "Modesty": trained_classifiers["Modesty"].predict([[feature_values["Letter Size"][0], feature_values["Top Margin"][0]]])[0],
            "Personal Harmony/Flexibility": trained_classifiers["Personal Harmony/Flexibility"].predict([[feature_values["Line Spacing"][0], feature_values["Word Spacing"][0]]])[0],
            "Lack of Discipline": trained_classifiers["Lack of Discipline"].predict([[feature_values["Slant Angle"][0], feature_values["Top Margin"][0]]])[0],
            "Poor Concentration": trained_classifiers["Poor Concentration"].predict([[feature_values["Letter Size"][0], feature_values["Line Spacing"][0]]])[0],
            "Non-communicativeness": trained_classifiers["Non-communicativeness"].predict([[feature_values["Letter Size"][0], feature_values["Word Spacing"][0]]])[0],
            "Social Isolation": trained_classifiers["Social Isolation"].predict([[feature_values["Line Spacing"][0], feature_values["Word Spacing"][0]]])[0],
        }

        for trait, prediction in predictions.items():
            print(f"✅ {trait}: {prediction}")

        print("---------------------------------------------------")

else:
    print("❌ Error: label_list.txt file not found.")
