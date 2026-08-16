'''Display students who scored more than 85 in Science.'''

import pandas as pd


def main():

    data = {
        'Name': ['Amit', 'Sagar', 'Pooja'],
        'Math': [85, 90, 78],
        'Science': [92, 88, 80],
        'English': [75, 85, 82]
    }

    df = pd.DataFrame(data)

    result = df[df['Science'] > 85]

    print("Students scoring more than 85 in Science:")
    print(result)


if __name__ == "__main__":
    main()