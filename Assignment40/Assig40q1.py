"""After training the Decision Tree model, use:
model.feature_importances_
Display importance score of each feature.
Which feature contributes the most in predicting FinalResult?
Which feature contributes the least?"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

Border = "_" * 40

def main():

    print(Border)
    print("Feature Importance")
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
    # Create and Train Model
    ################################################

    model = DecisionTreeClassifier()

    model.fit(X_train, Y_train)

    ################################################
    # Feature Importance
    ################################################

    Importance = model.feature_importances_

    print("\nFeature Importance Scores\n")

    for Feature, Score in zip(FeatureColumns, Importance):
        print("%-22s %.4f" % (Feature, Score))

    ################################################
    # Most and Least Important Feature
    ################################################

    MaxIndex = Importance.argmax()

    MinIndex = Importance.argmin()

    print("\nMost Important Feature :", FeatureColumns[MaxIndex])

    print("Least Important Feature :", FeatureColumns[MinIndex])


if __name__ == "__main__":
    main()