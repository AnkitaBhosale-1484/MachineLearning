import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def Regression(DataPath):

    Border = "-" * 40

    # Step 1 : Get Data
    print(Border)
    print("Step 1 : Get Data")
    print(Border)

    df = pd.read_csv(DataPath)

    print(df)


    # Step 2 : Clean, Prepare and Manipulate Data
    print(Border)
    print("Step 2 : Clean, Prepare and Manipulate Data")
    print(Border)

    X = df[["TV", "radio", "newspaper"]]
    Y = df["sales"]

    print("Independent Variables :")
    print(X)

    print("Dependent Variable :")
    print(Y)


    # Step 3 : Train Data
    print(Border)
    print("Step 3 : Train Data")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.5,
        random_state=42
    )

    print("Training Data :")
    print(X_train)

    print("Training Target :")
    print(Y_train)

    model = LinearRegression()

    model.fit(X_train, Y_train)

    print("Model trained successfully...")


    # Step 4 : Test the Data
    print(Border)
    print("Step 4 : Test the Data")
    print(Border)

    Y_pred = model.predict(X_test)


    # Step 5 : Display Predicted and Expected Values
    print(Border)
    print("Step 5 : Predicted and Expected Values")
    print(Border)

    print("Predicted Values :")
    print(Y_pred)

    print("\nExpected Values :")
    print(Y_test.values)

    print("\nComparison :")

    for predicted, expected in zip(Y_pred, Y_test):
        print("Predicted :", round(predicted, 2),
              "Expected :", expected)


def main():
    Regression("Advertising.csv")


if __name__ == "__main__":
    main()