'''Count how many students passed.'''

import pandas as pd


def main():

    data = {
        'Name': ['Amit', 'Sagar', 'Pooja'],
        'Math': [85, 90, 78],
        'Science': [92, 88, 80],
        'English': [75, 85, 82]
    }

    df = pd.DataFrame(data)

    df['Total'] = df['Math'] + df['Science'] + df['English']

    df['Status'] = df['Total'].apply(
        lambda x: 'Pass' if x >= 250 else 'Fail'
    )

    count = (df['Status'] == 'Pass').sum()

    print("Number of students passed:", count)


if __name__ == "__main__":
    main()