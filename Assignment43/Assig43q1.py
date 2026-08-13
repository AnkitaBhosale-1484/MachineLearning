import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def PlayPredictor():

    border = "_" * 40

    # Step 1: Get Data

    Data = pd.read_csv(
        "PlayPredictor.csv",
        index_col=0
    )

    # CSV has "Wether", assignment uses "Weather"
    Data.rename(
        columns={"Wether": "Weather"},
        inplace=True
    )

    # Step 2: Clean, Prepare and Manipulate Data

    WeatherEncoder = LabelEncoder()
    TemperatureEncoder = LabelEncoder()
    PlayEncoder = LabelEncoder()

    Data["Weather"] = WeatherEncoder.fit_transform(
        Data["Weather"]
    )

    Data["Temperature"] = TemperatureEncoder.fit_transform(
        Data["Temperature"]
    )

    Data["Play"] = PlayEncoder.fit_transform(
        Data["Play"]
    )

    # Features and Target

    X = Data[["Weather", "Temperature"]]
    Y = Data["Play"]

    # Step 3: Train Data

    K = 3

    Classifier = KNeighborsClassifier(
        n_neighbors=K
    )

    # Train using whole dataset

    Classifier.fit(X, Y)

    # Step 4: Test Data

    print(border)

    Weather = input("Enter Weather: ")
    Temperature = input("Enter Temperature: ")

    WeatherValue = WeatherEncoder.transform(
        [Weather]
    )[0]

    TemperatureValue = TemperatureEncoder.transform(
        [Temperature]
    )[0]

    TestData = pd.DataFrame(
        [[WeatherValue, TemperatureValue]],
        columns=["Weather", "Temperature"]
    )

    Result = Classifier.predict(TestData)

    ResultLabel = PlayEncoder.inverse_transform(
        Result
    )

    print(border)
    print("Predicted Result:", ResultLabel[0])

    # Step 5: Calculate Accuracy

    CheckAccuracy(X, Y)


def CheckAccuracy(X, Y):

    border = "_" * 40

    print(border)
    print("Accuracy for different values of K")
    print(border)

    # Divide dataset into two equal parts

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.5,
        random_state=42
    )

    # Calculate accuracy for different K values

    for K in [1, 3, 5]:

        Classifier = KNeighborsClassifier(
            n_neighbors=K
        )

        Classifier.fit(X_train, Y_train)

        Y_prediction = Classifier.predict(X_test)

        Accuracy = accuracy_score(
            Y_test,
            Y_prediction
        )

        print(
            "K =", K,
            "Accuracy =", Accuracy
        )


def main():
    PlayPredictor()


if __name__ == "__main__":
    main()