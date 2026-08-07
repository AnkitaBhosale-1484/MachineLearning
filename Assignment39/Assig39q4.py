"""Generate confusion matrix using sklearn.
Display it using ConfusionMatrixDisplay.
Explain clearly:
True Positive
True Negative
False Positive
False Negative"""

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

Border = "_" * 40

def main():

    print(Border)
    print("Confusion Matrix")
    print(Border)

    ################################################
    # Step 1 : Load Dataset
    ################################################

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    print("Dataset loaded successfully")

    ################################################
    # Step 2 : Features and Target
    ################################################

    FeatureColumns = [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted",
        "SleepHours"
    ]

    X = df[FeatureColumns]

    Y = df["FinalResult"]

    ################################################
    # Step 3 : Split Dataset
    ################################################

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    ################################################
    # Step 4 : Create Model
    ################################################

    model = DecisionTreeClassifier(random_state=42)

    ################################################
    # Step 5 : Train Model
    ################################################

    model.fit(X_train, Y_train)

    ################################################
    # Step 6 : Prediction
    ################################################

    Y_pred = model.predict(X_test)

    ################################################
    # Step 7 : Accuracy
    ################################################

    Accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy : %.2f%%" % (Accuracy * 100))

    ################################################
    # Step 8 : Confusion Matrix
    ################################################

    cm = confusion_matrix(Y_test, Y_pred)

    print("\nConfusion Matrix")
    print(cm)

    ################################################
    # Step 9 : Classification Report
    ################################################

    print("\nClassification Report")
    print(classification_report(Y_test, Y_pred))

    ################################################
    # Step 10 : Display Confusion Matrix
    ################################################

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail","Pass"]
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.show()


if __name__ == "__main__":
    main()