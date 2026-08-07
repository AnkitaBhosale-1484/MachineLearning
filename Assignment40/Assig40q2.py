"""Remove the column SleepHours from the dataset.
Train the model again.
Compare new accuracy with previous accuracy.
Does removing this feature affect performance?"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def main():

    print(Border)
    print("Remove SleepHours Feature")
    print(Border)

    ################################################
    # Load Dataset
    ################################################

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    ################################################
    # Model with All Features
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

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    Accuracy1 = accuracy_score(Y_test, Y_pred)

    ################################################
    # Model without SleepHours
    ################################################

    FeatureColumns = [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted"
    ]

    X = df[FeatureColumns]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    Accuracy2 = accuracy_score(Y_test, Y_pred)

    ################################################
    # Display Result
    ################################################

    print("Accuracy with SleepHours    : %.2f%%" % (Accuracy1 * 100))

    print("Accuracy without SleepHours : %.2f%%" % (Accuracy2 * 100))

    if Accuracy1 == Accuracy2:
        print("\nObservation : Removing SleepHours does not affect model performance.")

    elif Accuracy2 < Accuracy1:
        print("\nObservation : Accuracy decreased after removing SleepHours.")

    else:
        print("\nObservation : Accuracy increased after removing SleepHours.")


if __name__ == "__main__":
    main()