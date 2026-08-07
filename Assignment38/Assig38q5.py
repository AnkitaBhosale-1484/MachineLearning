"""Based on the dataset values, analyze whether:
Higher StudyHours increase the chance of passing.
Higher Attendance improves FinalResult.
Write your observations in 4–5 lines."""



import pandas as pd

Border = "_" * 30

def main():

    print(Border)
    print("Analysis")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    PassStudy = df[df["FinalResult"] == 1]["StudyHours"].mean()
    FailStudy = df[df["FinalResult"] == 0]["StudyHours"].mean()

    PassAttendance = df[df["FinalResult"] == 1]["Attendance"].mean()
    FailAttendance = df[df["FinalResult"] == 0]["Attendance"].mean()

    print("Average Study Hours (Pass) :", PassStudy)
    print("Average Study Hours (Fail) :", FailStudy)

    print("Average Attendance (Pass) :", PassAttendance)
    print("Average Attendance (Fail) :", FailAttendance)

    print("\nObservation:")

    if PassStudy > FailStudy:
        print("Higher StudyHours increase the chance of passing.")

    if PassAttendance > FailAttendance:
        print("Higher Attendance improves FinalResult.")

if __name__ == "__main__":
    main()