"""Create a plot showing the relationship between AssignmentsCompleted and FinalResult.
Explain your observation."""


import pandas as pd
import matplotlib.pyplot as plt

Border = "_" * 30

def main():

    print(Border)
    print("AssignmentsCompleted Analysis")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    Pass = df[df["FinalResult"] == 1]
    Fail = df[df["FinalResult"] == 0]

    plt.figure(figsize=(7,5))

    plt.scatter(Pass["AssignmentsCompleted"],
                Pass["FinalResult"],
                color="green",
                label="Pass")

    plt.scatter(Fail["AssignmentsCompleted"],
                Fail["FinalResult"],
                color="red",
                label="Fail")

    plt.title("AssignmentsCompleted vs FinalResult")

    plt.xlabel("AssignmentsCompleted")
    plt.ylabel("FinalResult")

    plt.legend()

    plt.grid()

    plt.show()

    print("Observation:")
    print("Students completing more assignments tend to perform better.")

if __name__ == "__main__":
    main()