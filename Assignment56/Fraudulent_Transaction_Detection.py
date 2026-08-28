import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = pd.read_csv("Fraudulent_Transaction_Detection.csv")

print("Dataset:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nFraud Distribution:")
print(data["Fraud"].value_counts())


# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

X = data.drop("Fraud", axis=1)
y = data["Fraud"]


# --------------------------------------------------
# 3. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)


# --------------------------------------------------
# 4. Create Models
# --------------------------------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42
)

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

adaboost = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 5. Voting Classifier
# --------------------------------------------------

voting = VotingClassifier(
    estimators=[
        ("Decision Tree", decision_tree),
        ("Random Forest", random_forest),
        ("AdaBoost", adaboost)
    ],
    voting="soft"
)


# --------------------------------------------------
# 6. Store All Models
# --------------------------------------------------

models = {
    "Decision Tree": decision_tree,
    "Bagging": bagging,
    "Random Forest": random_forest,
    "AdaBoost": adaboost,
    "Voting": voting
}


# --------------------------------------------------
# 7. Train and Evaluate
# --------------------------------------------------

results = []


for name, model in models.items():

    print("\n========================================")
    print(name)
    print("========================================")

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)


    # Print results
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:")
    print(cm)


    # Store results
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"]
    )

    disp.plot()

    plt.title(name + " - Confusion Matrix")
    plt.show()


# --------------------------------------------------
# 8. Final Comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\nFINAL COMPARISON")
print("==============================")

print(results_df.to_string(index=False))