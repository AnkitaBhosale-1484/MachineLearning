"""Create a scatter plot of:
StudyHours vs PreviousScore
Use different colors for Pass and Fail students."""


import pandas as pd
import matplotlib.pyplot as plt

Border = "_" * 30

def main():

    print(Border)
    print("Scatter Plot")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    plt.figure(figsize=(7,5))

    Pass = df[df["FinalResult"] == 1]
    Fail = df[df["FinalResult"] == 0]

    plt.scatter(Pass["StudyHours"],
                Pass["PreviousScore"],
                color="green",
                label="Pass")

    plt.scatter(Fail["StudyHours"],
                Fail["PreviousScore"],
                color="red",
                label="Fail")

    plt.title("StudyHours vs PreviousScore")

    plt.xlabel("StudyHours")
    plt.ylabel("PreviousScore")

    plt.legend()

    plt.grid()

    plt.show()

    print("Observation:")
    print("Students with higher StudyHours generally have better PreviousScore.")

if __name__ == "__main__":
    main()