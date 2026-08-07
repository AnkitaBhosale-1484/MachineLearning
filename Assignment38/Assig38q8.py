"""Draw a boxplot for Attendance.
Identify if any outliers are present."""


import pandas as pd
import matplotlib.pyplot as plt

Border = "_" * 30

def main():

    print(Border)
    print("Boxplot")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    plt.figure(figsize=(6,5))

    plt.boxplot(df["Attendance"])

    plt.title("Attendance Boxplot")
    plt.ylabel("Attendance")

    plt.grid()

    plt.show()

    print("Observation:")
    print("Points outside the whiskers are considered outliers.")

if __name__ == "__main__":
    main()