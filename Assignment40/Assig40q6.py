"""Identify students where:
Y_test != Y_pred
Display those rows.
How many students were misclassified?
What common pattern do you observe?"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

Border = "_" * 40

def main():

    print(Border)
    print("Misclassified Students")
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
    # Find Misclassified Students
    ################################################

    Misclassified = X_test.copy()

    Misclassified["Actual"] = Y_test.values

    Misclassified["Predicted"] = Y_pred

    Wrong = Misclassified[Misclassified["Actual"] != Misclassified["Predicted"]]

    ################################################
    # Display Result
    ################################################

    print("\nMisclassified Students\n")

    print(Wrong)

    print("\nTotal Misclassified Students :", len(Wrong))

    if len(Wrong) == 0:

        print("\nObservation :")
        print("No students were misclassified.")
        print("The model predicted all students correctly.")

    else:

        print("\nObservation :")
        print("Some students were classified incorrectly.")
        print("These students may have feature values")
        print("that are close to the decision boundary.")


if __name__ == "__main__":
    main()