
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("IRIS FLOWER DATASET")
print(X.head())
print("\nDataset Shape:", X.shape)
print("\nFlower Classes:", data.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

base_models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("KNN", KNeighborsClassifier(n_neighbors=5))
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000)
)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

voting_model = VotingClassifier(
    estimators=base_models,
    voting="hard"
)

models = [
    ("Logistic Regression", base_models[0][1]),
    ("Decision Tree", base_models[1][1]),
    ("KNN", base_models[2][1]),
    ("Stacking", stacking_model),
    ("AdaBoost", adaboost_model),
    ("Voting", voting_model)
]

results = []

for name, model in models:
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("\n", name)
    print(classification_report(
        y_test,
        predictions,
        target_names=data.target_names,
        zero_division=0
    ))

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.round(4).to_string(index=False))

best_model = results_df.loc[
    results_df["Accuracy"].idxmax(), "Model"
]

print("\nBEST MODEL BASED ON ACCURACY:", best_model)

results_df.set_index("Model")[[
    "Accuracy", "Precision", "Recall", "F1 Score"
]].plot(kind="bar", figsize=(12, 6))

plt.title("Base Models and Ensemble Models Performance")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=30)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nEnsemble Learning Implementation Completed")
