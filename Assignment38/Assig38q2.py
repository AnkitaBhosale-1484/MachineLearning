"""Write a program to:
Display total number of students in the dataset
Count how many students Passed (FinalResult = 1)
Count how many students Failed (FinalResult = 0)"""




import pandas as pd

Border = "_" * 30

def main():

    print(Border)
    print("Step 2 : Student Analysis")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    TotalStudents = len(df)

    PassedStudents = len(df[df["FinalResult"] == 1])

    FailedStudents = len(df[df["FinalResult"] == 0])

    print("Total Students :", TotalStudents)
    print("Passed Students :", PassedStudents)
    print("Failed Students :", FailedStudents)

if __name__ == "__main__":
    main()