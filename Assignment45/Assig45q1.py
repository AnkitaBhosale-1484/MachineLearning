'''Normalize the 'Math' scores using Min-Max scaling.'''

import pandas as pd


def main():

    data = {
        'Name': ['Amit', 'Sagar', 'Pooja'],
        'Math': [85, 90, 78],
        'Science': [92, 88, 80],
        'English': [75, 85, 82]
    }

    df = pd.DataFrame(data)

    df['Math'] = (df['Math'] - df['Math'].min()) / (
        df['Math'].max() - df['Math'].min()
    )

    print(df)


if __name__ == "__main__":
    main()