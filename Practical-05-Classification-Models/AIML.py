import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, ConfusionMatrixDisplay

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("CLASSIFICATION DATASET")
print(X.head())
print("\nDataset Shape:", X.shape)
print("\nClasses:", data.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": GaussianNB()
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    loss = log_loss(y_test, y_prob)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        loss
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Loss"]
)

print("\nCLASSIFICATION MODEL PERFORMANCE")
print(results_df.round(4).to_string(index=False))

best_model_name = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]

print("\nBEST MODEL BASED ON F1 SCORE:", best_model_name)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

results_df.set_index("Model")[metrics].plot(
    kind="bar", figsize=(10, 6)
)

plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Loss"])
plt.title("Log Loss Comparison")
plt.xlabel("Classification Model")
plt.ylabel("Log Loss")
plt.xticks(rotation=15)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

best_model = models[best_model_name]

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions[best_model_name],
    display_labels=data.target_names
)

plt.title("Confusion Matrix - " + best_model_name)
plt.tight_layout()
plt.show()

print("\nSAMPLE CLASSIFICATION PREDICTIONS")

sample = X_test.iloc[:5]
sample_predictions = best_model.predict(sample)

for i, prediction in enumerate(sample_predictions):
    print(
        "Sample", i + 1,
        "Predicted Class:", data.target_names[prediction],
        "Actual Class:", data.target_names[y_test[i]]
    )

print("\nEvaluation Completed")
