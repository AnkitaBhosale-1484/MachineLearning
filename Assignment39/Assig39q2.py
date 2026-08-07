"""Use the trained model to predict results for X_test.
Display predicted values along with actual values."""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

Border = "_" * 40

def main():

    print(Border)
    print("Student Performance Prediction")
    print(Border)

    ################################################
    # Step 1 : Load Dataset
    ################################################

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    print("Dataset loaded successfully")

    ################################################
    # Step 2 : Select Features and Target
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

    model = DecisionTreeClassifier()

    print("Model created successfully")

    ################################################
    # Step 5 : Train Model
    ################################################

    model.fit(X_train, Y_train)

    print("Model trained successfully")

    ################################################
    # Step 6 : Predict Result
    ################################################

    Y_pred = model.predict(X_test)

    print("\nActual Values")
    print(Y_test.values)

    print("\nPredicted Values")
    print(Y_pred)

    ################################################
    # Step 7 : Display Actual vs Predicted
    ################################################

    print("\nActual\tPredicted")
    print("-" * 25)

    for Actual, Predicted in zip(Y_test.values, Y_pred):
        print(Actual, "\t", Predicted)


if __name__ == "__main__":
    main()