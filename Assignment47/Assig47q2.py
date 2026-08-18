'''Using the regression model created in the previous question, 
write a Python program to predict marks for 6 study hours and display the predicted value.'''

import numpy as np
from sklearn.linear_model import LinearRegression

def main():

    # Dataset
    X = np.array([[1], [2], [3], [4], [5]])
    Y = np.array([50, 55, 60, 65, 70])

    # Create and train model
    model = LinearRegression()
    model.fit(X, Y)

    # Predict marks for 6 study hours
    prediction = model.predict([[6]])

    print("Predicted Marks for 6 Study Hours:", prediction[0])


if __name__ == "__main__":
    main()