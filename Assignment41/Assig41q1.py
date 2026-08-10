import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():

    # Step 1: Get Data
    data = pd.read_csv("WinePredictor.csv")

    print(data.head())
    print(data.columns)

    # Features and Target
    X = data.drop("Class", axis=1)
    Y = data["Class"]

    # Step 2: Clean, Prepare and Manipulate Data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=42
    )

    # Step 3: Train Data
    model = DecisionTreeClassifier()

    model = model.fit(X_train, Y_train)

    # Step 4: Test Data
    Y_pred = model.predict(X_test)

    # Step 5: Calculate Accuracy
    result = accuracy_score(Y_test, Y_pred)

    print("Accuracy is : ", result * 100)


if __name__ == "__main__":
    main()