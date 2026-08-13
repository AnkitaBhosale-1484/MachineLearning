'''Effect of K on KNN Prediction
The value of K plays an important role in the KNN algorithm.
Write a Python program that demonstrates how prediction changes when K changes.
Dataset
Use the same dataset as Assignment 1.
Tasks
Predict the class of the same new point using:
K = 1
K = 3
K = 5
Expected Output
Prediction Results
K = 1 → Red
K = 3 → Red
K = 5 → Blue
Explanation
Explain why the prediction changes when K increases.'''

import math

def MarvellousEucDistance(P1, P2):
    Ans = math.sqrt(
        (P1['X'] - P2['X']) ** 2 +
        (P1['Y'] - P2['Y']) ** 2
    )

    return Ans


def MarvellousKnnClassifier():

    Data = [
        {'point': 'A', 'X': 1, 'Y': 2, 'label': 'Red'},
        {'point': 'B', 'X': 2, 'Y': 3, 'label': 'Red'},
        {'point': 'C', 'X': 3, 'Y': 1, 'label': 'Blue'},
        {'point': 'D', 'X': 6, 'Y': 5, 'label': 'Blue'}
    ]

    # Same new point as Q1
    new_point = {'X': 2, 'Y': 2}

    # Calculate distance
    for d in Data:
        d['distance'] = MarvellousEucDistance(d, new_point)

    # Sort distances
    sorted_data = sorted(
        Data,
        key=lambda item: item['distance']
    )

    print("Prediction Results")

    # K = 1
    k = 1

    nearest = sorted_data[:k]

    votes = {}

    for neighbors in nearest:
        label = neighbors['label']
        votes[label] = votes.get(label, 0) + 1

    iMax = 0
    Name = ""

    for d in votes:
        if votes[d] > iMax:
            iMax = votes[d]
            Name = d

    print("K =", k, "→", Name)

    # K = 3
    k = 3

    nearest = sorted_data[:k]

    votes = {}

    for neighbors in nearest:
        label = neighbors['label']
        votes[label] = votes.get(label, 0) + 1

    iMax = 0
    Name = ""

    for d in votes:
        if votes[d] > iMax:
            iMax = votes[d]
            Name = d

    print("K =", k, "→", Name)

    # K = 5
    k = 5

    nearest = sorted_data[:k]

    votes = {}

    for neighbors in nearest:
        label = neighbors['label']
        votes[label] = votes.get(label, 0) + 1

    iMax = 0
    Name = ""

    for d in votes:
        if votes[d] > iMax:
            iMax = votes[d]
            Name = d

    print("K =", k, "→", Name)


def main():
    MarvellousKnnClassifier()


if __name__ == "__main__":
    main()


    '''K represents the number of nearest neighbors considered by the KNN algorithm. 
    When K is small, the prediction mainly depends on the closest data points.
      When K increases, more data points participate in voting. Therefore, 
      the majority class may change, and the final prediction can also change.'''