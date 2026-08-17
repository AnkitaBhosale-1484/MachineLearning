'''Plot a histogram of Math marks.'''

import pandas as pd
import matplotlib.pyplot as plt


def main():

    data = {
        'Name': ['Amit', 'Sagar', 'Pooja'],
        'Math': [85, 90, 78],
        'Science': [92, 88, 80],
        'English': [75, 85, 82]
    }

    df = pd.DataFrame(data)

    plt.hist(df['Math'], bins=5)

    plt.xlabel("Math Marks")
    plt.ylabel("Number of Students")
    plt.title("Distribution of Math Marks")

    plt.show()


if __name__ == "__main__":
    main()