'''write a python program using LinearRegression to train a regression model using the dataset below.
Study Hours	Marks
1	50
2	55
3	60
4	65
5	70
Your program should:
Train the regression model
Print the coefficient
Print the intercept'''

import numpy as np
from sklearn.linear_model import LinearRegression

def main():

    # Dataset
    X = np.array([[1], [2], [3], [4], [5]])
    Y = np.array([50, 55, 60, 65, 70])

    # Create Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X, Y)

    # Print coefficient
    print("Coefficient:", model.coef_[0])

    # Print intercept
    print("Intercept:", model.intercept_)


if __name__ == "__main__":
    main()