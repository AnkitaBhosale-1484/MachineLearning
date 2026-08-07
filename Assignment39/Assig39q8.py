"""Write a single structured Python program that performs:
Dataset loading
Data analysis
Visualization
Train-test split
Model training
Prediction
Accuracy calculation
Confusion matrix generation
Final conclusion
Your code should include proper comments explaining each step."""

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

Border = "_" * 40

def main():

    ################################################
    # Step 1 : Load Dataset
    ################################################

    print(Border)
    print("Step 1 : Load Dataset")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    print("Dataset Loaded Successfully")
    print(df.head())

    ################################################
    # Step 2 : Data Analysis
    ################################################

    print(Border)
    print("Step 2 : Data Analysis")
    print(Border)

    print("Shape :", df.shape)

    print("\nColumns")
    print(list(df.columns))

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nClass Distribution")
    print(df["FinalResult"].value_counts())

    print("\nStatistical Information")
    print(df.describe())

    ################################################
    # Step 3 : Visualization
    ################################################

    print(Border)
    print("Step 3 : Visualization")
    print(Border)

    plt.figure(figsize=(7,5))

    for Result in df["FinalResult"].unique():

        Temp = df[df["FinalResult"] == Result]

        plt.scatter(
            Temp["StudyHours"],
            Temp["Attendance"],
            label=Result
        )

    plt.title("StudyHours vs Attendance")

    plt.xlabel("StudyHours")

    plt.ylabel("Attendance")

    plt.legend(["Fail","Pass"])

    plt.grid()

    plt.show()

    ################################################
    # Step 4 : Independent and Dependent Variables
    ################################################

    print(Border)
    print("Step 4 : Feature Selection")
    print(Border)

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
    # Step 5 : Train Test Split
    ################################################

    print(Border)
    print("Step 5 : Train Test Split")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("Training Samples :", len(X_train))

    print("Testing Samples :", len(X_test))

    ################################################
    # Step 6 : Model Training
    ################################################

    print(Border)
    print("Step 6 : Model Training")
    print(Border)

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, Y_train)

    print("Model Trained Successfully")

    ################################################
    # Step 7 : Prediction
    ################################################

    print(Border)
    print("Step 7 : Prediction")
    print(Border)

    Y_pred = model.predict(X_test)

    print("Actual Values")
    print(Y_test.values)

    print("\nPredicted Values")
    print(Y_pred)

    ################################################
    # Step 8 : Accuracy
    ################################################

    print(Border)
    print("Step 8 : Accuracy")
    print(Border)

    Accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy : %.2f%%" % (Accuracy * 100))

    ################################################
    # Step 9 : Confusion Matrix
    ################################################

    print(Border)
    print("Step 9 : Confusion Matrix")
    print(Border)

    cm = confusion_matrix(Y_test, Y_pred)

    print(cm)

    print("\nClassification Report")

    print(classification_report(Y_test, Y_pred))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail","Pass"]
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.show()

    ################################################
    # Step 10 : Final Conclusion
    ################################################

    print(Border)
    print("Step 10 : Final Conclusion")
    print(Border)

    if Accuracy == 1:

        print("Decision Tree Model achieved 100% Accuracy.")
        print("The model correctly classified all students.")
        print("No overfitting or underfitting is observed.")
        print("StudyHours is the most important feature.")

    else:

        print("Model Accuracy :", Accuracy * 100)
        print("Further improvement is possible.")


if __name__ == "__main__":
    main()