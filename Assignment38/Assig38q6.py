"""Plot a histogram of StudyHours.
Explain what the distribution tells you."""


import pandas as pd
import matplotlib.pyplot as plt

Border = "_" * 30

def main():

    print(Border)
    print("Histogram of StudyHours")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    plt.figure(figsize=(7,5))

    plt.hist(df["StudyHours"], bins=10, edgecolor="black")

    plt.title("Histogram of StudyHours")
    plt.xlabel("StudyHours")
    plt.ylabel("Frequency")

    plt.grid()

    plt.show()

    print("Observation:")
    print("Histogram shows the distribution of StudyHours.")

if __name__ == "__main__":
    main()