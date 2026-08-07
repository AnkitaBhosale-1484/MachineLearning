"""Use value_counts() to analyze the distribution of FinalResult.

Calculate the percentage of Pass and Fail students.
Is the dataset balanced? Justify your answer."""


import pandas as pd

Border = "_" * 30

def main():

    print(Border)
    print("FinalResult Analysis")
    print(Border)

    DataPath = "student_performance_ml.csv"

    df = pd.read_csv(DataPath)

    ResultCount = df["FinalResult"].value_counts()

    print("Pass / Fail Count")
    print(ResultCount)

    Percentage = (ResultCount / len(df)) * 100

    print("\nPercentage")
    print(Percentage)

    if abs(Percentage[1] - Percentage[0]) <= 10:
        print("\nDataset is Balanced")
    else:
        print("\nDataset is Imbalanced")

if __name__ == "__main__":
    main()