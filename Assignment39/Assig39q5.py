"""Calculate:
Training Accuracy
Testing Accuracy
Compare both and comment whether the model is overfitting or underfitting."""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def main():

    print(Border)
    print("Training and Testing Accuracy")
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
    # Step 6 : Training Prediction
    ################################################

    Y_train_pred = model.predict(X_train)

    ################################################
    # Step 7 : Testing Prediction
    ################################################

    Y_test_pred = model.predict(X_test)

    ################################################
    # Step 8 : Accuracy Calculation
    ################################################

    TrainAccuracy = accuracy_score(Y_train, Y_train_pred)

    TestAccuracy = accuracy_score(Y_test, Y_test_pred)

    ################################################
    # Step 9 : Display Accuracy
    ################################################

    print("Training Accuracy : %.2f%%" % (TrainAccuracy * 100))
    print("Testing Accuracy  : %.2f%%" % (TestAccuracy * 100))

    ################################################
    # Step 10 : Observation
    ################################################

    if TrainAccuracy == 1.0 and TestAccuracy == 1.0:

        print("\nObservation :")
        print("The model is performing well.")
        print("No overfitting or underfitting is observed.")

    elif TrainAccuracy > TestAccuracy:

        print("\nObservation :")
        print("The model is overfitting.")
        print("Training accuracy is higher than testing accuracy.")

    elif TrainAccuracy < TestAccuracy:

        print("\nObservation :")
        print("The model is underfitting.")

    else:

        print("\nObservation :")
        print("Training and testing accuracy are almost the same.")


if __name__ == "__main__":
    main()