"""Create a new column:
PerformanceIndex = (StudyHours * 2) + Attendance
Train the model including this new feature.
Does accuracy improve?"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def main():

    print(Border)
    print("PerformanceIndex Feature")
    print(Border)

    ################################################
    # Load Dataset
    ################################################

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    ################################################
    # Create New Feature
    ################################################

    df["PerformanceIndex"] = (df["StudyHours"] * 2) + df["Attendance"]

    ################################################
    # Features and Target
    ################################################

    FeatureColumns = [
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted",
        "SleepHours",
        "PerformanceIndex"
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
    # Accuracy
    ################################################

    Accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy : %.2f%%" % (Accuracy * 100))

    ################################################
    # Display New Column
    ################################################

    print("\nPerformanceIndex")

    print(df[["StudyHours","Attendance","PerformanceIndex"]].head())

    ################################################
    # Observation
    ################################################

    if Accuracy == 1:
        print("\nObservation :")
        print("Accuracy did not improve because")
        print("the model was already achieving 100% accuracy.")
    else:
        print("\nObservation :")
        print("PerformanceIndex improved the model.")


if __name__ == "__main__":
    main()