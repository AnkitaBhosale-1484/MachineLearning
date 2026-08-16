'''Plot a line chart of marks for 'Amit' across all subjects.'''
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

    amit = df[df['Name'] == 'Amit'].iloc[0]

    subjects = ['Math', 'Science', 'English']
    marks = [amit['Math'], amit['Science'], amit['English']]

    plt.plot(subjects, marks, marker='o')

    plt.xlabel("Subjects")
    plt.ylabel("Marks")
    plt.title("Amit's Marks Across All Subjects")

    plt.show()


if __name__ == "__main__":
    main()