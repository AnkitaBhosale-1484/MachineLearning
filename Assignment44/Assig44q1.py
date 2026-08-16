'''Create a DataFrame for student marks and print basic information like shape, columns, and data types.'''
import pandas as pd

def main():

    data = {
        'Name': ['Amit', 'Sagar', 'Pooja'],
        'Math': [85, 90, 78],
        'Science': [92, 88, 80],
        'English': [75, 85, 82]
    }

    df = pd.DataFrame(data)

    print("DataFrame:")
    print(df)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)


if __name__ == "__main__":
    main()