import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
import joblib

# -----------------------------
# 1. Generate Bigger Dataset (1000 samples)
# -----------------------------

np.random.seed(42)

N = 1000

attendance = np.random.randint(30, 100, N)
marks = np.random.randint(20, 100, N)
income = np.random.randint(1, 6, N)
study_hours = np.random.randint(1, 6, N)
failures = np.random.randint(0, 5, N)

# Rule-based dropout generation (realistic logic)
dropout = []

for i in range(N):
    risk = 0

    if attendance[i] < 50:
        risk += 1
    if marks[i] < 50:
        risk += 1
    if failures[i] >= 2:
        risk += 1
    if study_hours[i] <= 2:
        risk += 1
    if income[i] <= 2:
        risk += 1

    # High risk -> dropout = 1
    if risk >= 3:
        dropout.append(1)
    else:
        dropout.append(0)

data = pd.DataFrame({
    "attendance": attendance,
    "marks": marks,
    "income": income,
    "study_hours": study_hours,
    "failures": failures,
    "dropout": dropout
})

data = shuffle(data)

print("Dataset size:", data.shape)
print(data["dropout"].value_counts())

# -----------------------------
# 2. Split features & labels
# -----------------------------

X = data.drop("dropout", axis=1)
y = data["dropout"]

# -----------------------------
# 3. Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. Train Better Random Forest
# -----------------------------

model = RandomForestClassifier(
    n_estimators=300,          # more trees
    max_depth=10,             # control overfitting
    min_samples_split=5,
    class_weight="balanced", # handle imbalance
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 5. Cross-Validation Score
# -----------------------------

cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print("Cross-validation accuracies:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# -----------------------------
# 6. Evaluate on Test Set
# -----------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nTest Accuracy:", accuracy * 100)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Save trained model
# -----------------------------

joblib.dump(model, "dropout_model.pkl")

print("\nModel saved as dropout_model.pkl")
