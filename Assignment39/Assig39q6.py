"""Train three Decision Tree models with:
max_depth = 1
max_depth = 3
max_depth = None
Compare their testing accuracies and write your observations."""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def main():

    print(Border)
    print("Decision Tree with Different Max Depth")
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
    # Step 4 : Model with max_depth = 1
    ################################################

    model1 = DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    )

    model1.fit(X_train, Y_train)

    Y_pred1 = model1.predict(X_test)

    Accuracy1 = accuracy_score(Y_test, Y_pred1)

    ################################################
    # Step 5 : Model with max_depth = 3
    ################################################

    model2 = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    model2.fit(X_train, Y_train)

    Y_pred2 = model2.predict(X_test)

    Accuracy2 = accuracy_score(Y_test, Y_pred2)

    ################################################
    # Step 6 : Model with max_depth = None
    ################################################

    model3 = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    model3.fit(X_train, Y_train)

    Y_pred3 = model3.predict(X_test)

    Accuracy3 = accuracy_score(Y_test, Y_pred3)

    ################################################
    # Step 7 : Display Accuracy
    ################################################

    print("Testing Accuracy with max_depth = 1    : %.2f%%" % (Accuracy1 * 100))

    print("Testing Accuracy with max_depth = 3    : %.2f%%" % (Accuracy2 * 100))

    print("Testing Accuracy with max_depth = None : %.2f%%" % (Accuracy3 * 100))

    ################################################
    # Step 8 : Observation
    ################################################

    print("\nObservation:")

    if Accuracy1 == Accuracy2 == Accuracy3:

        print("All three models give the same testing accuracy.")
        print("Increasing tree depth does not improve performance.")

    elif Accuracy3 > Accuracy2 and Accuracy2 > Accuracy1:

        print("Accuracy increases as max_depth increases.")

    else:

        print("Different max_depth values produce different accuracies.")


if __name__ == "__main__":
    main()