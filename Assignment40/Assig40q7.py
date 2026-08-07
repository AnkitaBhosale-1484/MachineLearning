"""Train model using:
random_state = 0
random_state = 10
random_state = 42
Compare testing accuracy.
Does the result change?"""




import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border = "_" * 40

def TrainModel(RandomValue, X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=RandomValue
    )

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    Accuracy = accuracy_score(Y_test, Y_pred)

    print("Random State :", RandomValue)
    print("Testing Accuracy : %.2f%%" % (Accuracy * 100))
    print()

def main():

    print(Border)
    print("Random State Comparison")
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
    # Train with Different Random States
    ################################################

    TrainModel(0, X, Y)

    TrainModel(10, X, Y)

    TrainModel(42, X, Y)

    print("Observation : Compare the testing accuracy for different random states.")

if __name__ == "__main__":
    main()