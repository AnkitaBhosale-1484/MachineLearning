'''Drop the 'English' column from original DataFrame.'''
import pandas as pd


def main():

    data = {
        'Name': ['Amit', 'Sagar', 'Pooja'],
        'Math': [85, 90, 78],
        'Science': [92, 88, 80],
        'English': [75, 85, 82]
    }

    df = pd.DataFrame(data)

    df = df.drop('English', axis=1)

    print(df)


if __name__ == "__main__":
    main()