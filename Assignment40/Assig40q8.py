"""Decision Tree Visualization
Use:
from sklearn.tree import plot_tree
Visualize the trained Decision Tree.
Which feature appears as the root node?
Why do you think that feature was selected first?"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

Border = "_" * 40

def main():

    print(Border)
    print("Decision Tree Visualization")
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
    # Plot Decision Tree
    ################################################

    plt.figure(figsize=(12,8))

    plot_tree(
        model,
        feature_names=FeatureColumns,
        class_names=["Fail","Pass"],
        filled=True,
        rounded=True
    )

    plt.title("Decision Tree")

    plt.show()

    ################################################
    # Observation
    ################################################

    print("\nRoot Node Feature : StudyHours")
    print("Reason : StudyHours gives the best split and")
    print("has the highest importance in this dataset.")


if __name__ == "__main__":
    main()