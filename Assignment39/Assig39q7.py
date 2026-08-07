"""Use the trained model to predict result for a student with:
StudyHours = 6
Attendance = 85
PreviousScore = 66
AssignmentsCompleted = 7
SleepHours = 7
Will the student Pass or Fail?"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

Border = "_" * 40

def main():

    print(Border)
    print("Predict Result for New Student")
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
    # Step 4 : Create and Train Model
    ################################################

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, Y_train)

    ################################################
    # Step 5 : New Student Data
    ################################################

    NewStudent = pd.DataFrame({
        "StudyHours": [6],
        "Attendance": [85],
        "PreviousScore": [66],
        "AssignmentsCompleted": [7],
        "SleepHours": [7]
    })

    ################################################
    # Step 6 : Prediction
    ################################################

    Prediction = model.predict(NewStudent)

    ################################################
    # Step 7 : Display Result
    ################################################

    print("\nStudent Details")
    print(NewStudent)

    if Prediction[0] == 1:
        print("\nPrediction : PASS")
    else:
        print("\nPrediction : FAIL")


if __name__ == "__main__":
    main()