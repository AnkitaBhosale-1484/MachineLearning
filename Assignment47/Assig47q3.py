'''Consider the dataset below:
StudyHours	SleepHours	Marks
1	7	50
2	6	55
3	7	60
4	6	65
5	8	70
Write a Python program to:
Train a regression model using this dataset
Print the coefficients for both features
Print the intercept'''

import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([
    [1, 7],
    [2, 6],
    [3, 7],
    [4, 6],
    [5, 8]
])

Y = np.array([50, 55, 60, 65, 70])

# Create regression model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Print coefficients
print("Coefficient for Study Hours:", model.coef_[0])
print("Coefficient for Sleep Hours:", model.coef_[1])

# Print intercept
print("Intercept:", model.intercept_)