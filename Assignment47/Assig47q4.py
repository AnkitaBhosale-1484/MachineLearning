import numpy as np
from sklearn.preprocessing import StandardScaler


def main():

    data = np.array([
        [25, 20000],
        [30, 40000],
        [35, 80000]
    ])

    point1 = data[0]
    point2 = data[1]

    # Distance before scaling
    distance_before = np.linalg.norm(point1 - point2)

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Distance after scaling
    distance_after = np.linalg.norm(
        scaled_data[0] - scaled_data[1]
    )

    print("Point 1 :", point1)
    print("Point 2 :", point2)

    print("Euclidean Distance Before Scaling :",
          distance_before)

    print("Euclidean Distance After Scaling :",
          distance_after)


if __name__ == "__main__":
    main()