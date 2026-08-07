"""Calculate model accuracy using accuracy_score.
Display the result in percentage format."""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def main():

    print(Border)
    print("Model Accuracy")
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
    # Step 4 : Create Model
    ################################################

    model = DecisionTreeClassifier(random_state=42)

    ################################################
    # Step 5 : Train Model
    ################################################

    model.fit(X_train, Y_train)

    ################################################
    # Step 6 : Prediction
    ################################################

    Y_pred = model.predict(X_test)

    ################################################
    # Step 7 : Calculate Accuracy
    ################################################

    Accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy of Model : %.2f%%" % (Accuracy * 100))


if __name__ == "__main__":
    main()