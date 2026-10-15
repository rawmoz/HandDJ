import csv

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset: first column is the label, the other 63 are the features
X, y = [], []
with open("gestures.csv") as f:
    for row in csv.reader(f):
        y.append(row[0])
        X.append([float(v) for v in row[1:]])
print(f"Loaded {len(X)} samples")

# Hide 20% of the data from the model so we can quiz it on samples it never saw
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train: 100 decision trees that each learn rules like "if the pinky is up and..."
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Quiz the model on the hidden 20%
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.1%}")

# Confusion matrix: rows = the real gesture, columns = what the model guessed
labels = sorted(set(y))
matrix = confusion_matrix(y_test, predictions, labels=labels)
print("\nConfusion matrix (rows = real, columns = guessed)")
print(" " * 10 + "".join(f"{label:>9}" for label in labels))
for label, row in zip(labels, matrix):
    print(f"{label:>10}" + "".join(f"{n:>9}" for n in row))

joblib.dump(model, "model.joblib")
print("\nSaved model to model.joblib")
