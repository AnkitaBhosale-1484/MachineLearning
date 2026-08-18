'''
Using the same dataset from above question, calculate model performance.
Tasks:
Predict all Y values using regression equation.
Calculate:
Mean Squared Error (MSE)
R² Score
Show all intermediate calculations.'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():

    # Dataset
    X = np.array([
        [1, 7],
        [2, 6],
        [3, 7],
        [4, 6],
        [5, 8]
    ])

    Y = np.array([50, 55, 60, 65, 70])

    # Create and train model
    model = LinearRegression()
    model.fit(X, Y)

    # Predict all Y values
    Y_pred = model.predict(X)

    print("Actual Y values:", Y)
    print("Predicted Y values:", Y_pred)

    # Intermediate calculations
    print("\nIntermediate Calculations:")

    for i in range(len(Y)):
        error = Y[i] - Y_pred[i]
        squared_error = error ** 2

        print("Actual:", Y[i],
              "Predicted:", Y_pred[i],
              "Error:", error,
              "Squared Error:", squared_error)

    # Calculate MSE
    mse = mean_squared_error(Y, Y_pred)

    # Calculate R2 Score
    r2 = r2_score(Y, Y_pred)

    print("\nMean Squared Error (MSE):", mse)
    print("R2 Score:", r2)


if __name__ == "__main__":
    main()
