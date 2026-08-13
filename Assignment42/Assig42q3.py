'''Student Pass/Fail Prediction using KNN
Use KNN to predict whether a student passes or fails based on study hours and attendance.
Dataset
Study Hours	Attendance	Result
2	60	Fail
5	80	Pass
6	85	Pass
1	50	Fail
Tasks
Accept input from the user:
Study hours
Attendance percentage
Apply KNN algorithm.
Predict whether the student Passes or Fails.
Input Example
Enter Study Hours: 4
Enter Attendance: 70
Expected Output
Predicted Result: Pass'''


import math

def MarvellousEucDistance(P1, P2):
    Ans = math.sqrt(
        (P1['StudyHours'] - P2['StudyHours']) ** 2 +
        (P1['Attendance'] - P2['Attendance']) ** 2
    )

    return Ans


def MarvellousKnnClassifier(k=3):

    Data = [
        {'StudyHours': 2, 'Attendance': 60, 'label': 'Fail'},
        {'StudyHours': 5, 'Attendance': 80, 'label': 'Pass'},
        {'StudyHours': 6, 'Attendance': 85, 'label': 'Pass'},
        {'StudyHours': 1, 'Attendance': 50, 'label': 'Fail'}
    ]

    StudyHours = int(input("Enter Study Hours: "))
    Attendance = int(input("Enter Attendance: "))

    new_point = {
        'StudyHours': StudyHours,
        'Attendance': Attendance
    }

    # Calculate distance
    for d in Data:
        d['distance'] = MarvellousEucDistance(d, new_point)

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
            d['label'],
            "- Distance:",
            round(d['distance'], 2)
        )

    # Voting
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

    print("\nPredicted Result:", Name)


def main():
    MarvellousKnnClassifier()


if __name__ == "__main__":
    main()