"""Train model with:
max_depth = None
Calculate:
Training accuracy
Testing accuracy
If training accuracy is 100% but testing accuracy is lower, explain why this happens."""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def main():

    print(Border)
    print("Decision Tree with max_depth = None")
    print(Border)

    ################################################
    # Step 1 : Load Dataset
    ################################################

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

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

    model = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

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
    # Step 8 : Accuracy
    ################################################

    TrainAccuracy = accuracy_score(Y_train, Y_train_pred)

    TestAccuracy = accuracy_score(Y_test, Y_test_pred)

    ################################################
    # Step 9 : Display Result
    ################################################

    print("Training Accuracy : %.2f%%" % (TrainAccuracy * 100))

    print("Testing Accuracy  : %.2f%%" % (TestAccuracy * 100))

    ################################################
    # Step 10 : Observation
    ################################################

    if TrainAccuracy == 1 and TestAccuracy < 1:

        print("\nObservation :")
        print("The model is overfitting.")
        print("It learned the training data perfectly")
        print("but could not generalize well on test data.")

    elif TrainAccuracy == 1 and TestAccuracy == 1:

        print("\nObservation :")
        print("Training and testing accuracy are both 100%.")
        print("No overfitting is observed on this dataset.")

    else:

        print("\nObservation :")
        print("The model is performing normally.")


if __name__ == "__main__":
    main()