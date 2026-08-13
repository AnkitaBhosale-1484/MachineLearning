'''Manual K-Nearest Neighbors Classification
Write a Python program that classifies a new data point using the K-Nearest Neighbors algorithm.
The algorithm should be implemented manually without using any machine learning library.
The program should:
Calculate Euclidean distance
Sort distances
Select K nearest neighbors
Predict the class based on majority voting
Dataset
Point	X	Y	Label
A	1	2	Red
B	2	3	Red
C	3	1	Blue
D	6	5	Blue
Tasks
Accept X and Y coordinates of a new point from the user.
Compute Euclidean distance from all dataset points.
Sort the distances.
Select K = 3 nearest neighbors.
Predict the class label.
Input Format
Enter X coordinate: 2
Enter Y coordinate: 2
Expected Output
Nearest Neighbors:
A - Distance: 1.0
B - Distance: 1.0
C - Distance: 1.41

Predicted Class: Red'''




import math

def EucDistance(P1, P2):
    Ans = math.sqrt(
        (P1['X'] - P2['X']) ** 2 +
        (P1['Y'] - P2['Y']) ** 2
    )

    return Ans


def KnnClassifier(k=3):

    border = "_" * 30

    Data = [
        {'point': 'A', 'X': 1, 'Y': 2, 'label': 'Red'},
        {'point': 'B', 'X': 2, 'Y': 3, 'label': 'Red'},
        {'point': 'C', 'X': 3, 'Y': 1, 'label': 'Blue'},
        {'point': 'D', 'X': 6, 'Y': 5, 'label': 'Blue'}
    ]

    X = int(input("Enter X coordinate: "))
    Y = int(input("Enter Y coordinate: "))

    new_point = {'X': X, 'Y': Y}

    # Calculate distance
    for d in Data:
        d['distance'] = EucDistance(d, new_point)

    # Sort distances
    sorted_data = sorted(
        Data,
        key=lambda item: item['distance']
    )

    # Select K nearest neighbors
    nearest = sorted_data[:k]

    print("\nNearest Neighbors:")

    for d in nearest:
        print(
            d['point'],
            "- Distance:",
            round(d['distance'], 2)
        )

    # Majority voting
    votes = {}

    for neighbors in nearest:
        label = neighbors['label']
        votes[label] = votes.get(label, 0) + 1

    # Find maximum votes
    iMax = 0
    Name = ""

    for d in votes:
        if votes[d] > iMax:
            iMax = votes[d]
            Name = d

    print("\nPredicted Class:", Name)


def main():
    KnnClassifier()


if __name__ == "__main__":
    main()
