"""Create a new DataFrame with details of 5 new students.
Use the trained model to predict their results.
Display predictions clearly."""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

Border = "_" * 40

def main():

    print(Border)
    print("Prediction for New Students")
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
    # Create New Students DataFrame
    ################################################

    NewStudents = pd.DataFrame({
        "StudyHours":[2,4,5,6,8],
        "Attendance":[65,75,82,88,95],
        "PreviousScore":[45,56,62,70,80],
        "AssignmentsCompleted":[3,5,6,7,9],
        "SleepHours":[5,6,7,8,8]
    })

    ################################################
    # Predict
    ################################################

    Prediction = model.predict(NewStudents)

    ################################################
    # Display Result
    ################################################

    NewStudents["Prediction"] = Prediction

    print(NewStudents)

    print("\nPrediction Meaning")

    for i in range(len(Prediction)):

        if Prediction[i] == 1:
            print("Student", i+1, ": Pass")

        else:
            print("Student", i+1, ": Fail")


if __name__ == "__main__":
    main()