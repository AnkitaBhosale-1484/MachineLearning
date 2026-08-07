"""Without using accuracy_score, manually calculate accuracy.
Verify whether it matches sklearn accuracy."""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def main():

    print(Border)
    print("Manual Accuracy Calculation")
    print(Border)

    ################################################
    # Load Dataset
    ################################################

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    ################################################
    # Features and Target
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
    # Split Dataset
    ################################################

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    ################################################
    # Train Model
    ################################################

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, Y_train)

    ################################################
    # Prediction
    ################################################

    Y_pred = model.predict(X_test)

    ################################################
    # Manual Accuracy
    ################################################

    Correct = 0

    for Actual, Predicted in zip(Y_test, Y_pred):

        if Actual == Predicted:
            Correct = Correct + 1

    Total = len(Y_test)

    ManualAccuracy = (Correct / Total) * 100

    ################################################
    # Sklearn Accuracy
    ################################################

    SklearnAccuracy = accuracy_score(Y_test, Y_pred) * 100

    ################################################
    # Display Result
    ################################################

    print("Correct Predictions :", Correct)
    print("Total Predictions   :", Total)

    print("\nManual Accuracy  : %.2f%%" % ManualAccuracy)
    print("Sklearn Accuracy : %.2f%%" % SklearnAccuracy)

    if round(ManualAccuracy, 2) == round(SklearnAccuracy, 2):
        print("\nObservation : Both accuracies are the same.")
    else:
        print("\nObservation : Both accuracies are different.")


if __name__ == "__main__":
    main()