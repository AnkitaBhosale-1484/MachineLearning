'''
Consider the following task:
Train a linear regression model.
Predict salary for 6 years of experience.
Plot the regression line using Matplotlib.
Dataset:
Experience	Salary
1	20000
2	25000
3	30000
4	35000
5	40000
Expected Output:
Predicted Salary for 6 Years Experience: ₹45000
Graph should display:
Data points
Regression line'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():

    # Dataset
    X = np.array([[1], [2], [3], [4], [5]])
    Y = np.array([20000, 25000, 30000, 35000, 40000])

    # Create and train model
    model = LinearRegression()
    model.fit(X, Y)

    # Predict salary for 6 years experience
    prediction = model.predict([[6]])

    print("Predicted Salary for 6 Years Experience: ₹", int(prediction[0]))

    # Regression line
    Y_pred = model.predict(X)

    # Plot data points
    plt.scatter(X, Y, label="Data Points")

    # Plot regression line
    plt.plot(X, Y_pred, label="Regression Line")

    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Experience vs Salary")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()