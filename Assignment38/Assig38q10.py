"""Plot SleepHours against FinalResult.
Does sleeping more guarantee success? Explain."""


import pandas as pd
import matplotlib.pyplot as plt

Border = "_" * 30

def main():

    print(Border)
    print("SleepHours Analysis")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    Pass = df[df["FinalResult"] == 1]
    Fail = df[df["FinalResult"] == 0]

    plt.figure(figsize=(7,5))

    plt.scatter(Pass["SleepHours"],
                Pass["FinalResult"],
                color="green",
                label="Pass")

    plt.scatter(Fail["SleepHours"],
                Fail["FinalResult"],
                color="red",
                label="Fail")

    plt.title("SleepHours vs FinalResult")

    plt.xlabel("SleepHours")
    plt.ylabel("FinalResult")

    plt.legend()

    plt.grid()

    plt.show()

    print("Observation:")
    print("Sleeping more alone does not guarantee success.")
    print("StudyHours, Attendance and PreviousScore also affect the FinalResult.")

if __name__ == "__main__":
    main()