"""Write a Python program to load the file student_performance_ml.csv using pandas.
Display:
First 5 records
Last 5 records
Total number of rows and columns
List of column names
Data types of each column"""


import pandas as pd



Border="_"*50


def main():
    print(Border)
    print("student performance case study")
    print(Border)

    DataPath = "student_performance_ml.csv"
    df=pd.read_csv(DataPath)

    print("Data set loaded sucessfully")
    print(Border)

    print("first five records")
    print(df.head())
    print(Border)


    print("Last 5 records")
    print(df.tail())
    print(Border)

    print("total number of rows and columns")
    print(df.shape)
    print(Border)

    print("list of column names")
    print(df.columns)
    print(Border)

    print("data types of each column")
    print(df.dtypes)
    print(Border)



if __name__=="__main__":
    main()