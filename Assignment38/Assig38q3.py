"""Using pandas functions, calculate and display:
Average StudyHours
Average Attendance
Maximum PreviousScore
Minimum SleepHours"""


import pandas as pd

Border = "_" * 30

def main():

    print(Border)
    print("Step 3 : Statistical Analysis")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    print("Average Study Hours :", df["StudyHours"].mean())

    print("Average Attendance :", df["Attendance"].mean())

    print("Maximum Previous Score :", df["PreviousScore"].max())

    print("Minimum Sleep Hours :", df["SleepHours"].min())

if __name__ == "__main__":
    main()