"""Train the model using only:
StudyHours
Attendance
Compare the accuracy with the full-feature model.
Is the model still performing well?"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def main():

    print(Border)
    print("Model with Limited Features")
    print(Border)

    ################################################
    # Load Dataset
    ################################################

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    Y = df["FinalResult"]

    ################################################
    # Full Feature Model
    ################################################

    FullFeatures = [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted",
        "SleepHours"
    ]

    X = df[FullFeatures]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    FullAccuracy = accuracy_score(Y_test, Y_pred)

    ################################################
    # Model with only 2 Features
    ################################################

    LimitedFeatures = [
        "StudyHours",
        "Attendance"
    ]

    X = df[LimitedFeatures]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    LimitedAccuracy = accuracy_score(Y_test, Y_pred)

    ################################################
    # Result
    ################################################

    print("Full Feature Accuracy    : %.2f%%" % (FullAccuracy * 100))

    print("Limited Feature Accuracy : %.2f%%" % (LimitedAccuracy * 100))

    if LimitedAccuracy >= FullAccuracy:
        print("\nObservation : Model is still performing well.")

    else:
        print("\nObservation : Performance decreased.")


if __name__ == "__main__":
    main()