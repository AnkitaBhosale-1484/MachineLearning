'''Write a Python program that calculates the variance and standard deviation of the dataset:
[6, 7, 8, 9, 10, 11, 12]

Display both results.'''

import numpy as np

def main():
    data=np.array([6,7,8,9,10,11,12])
    variance=np.var(data)
    standard_deviation=np.std(data)

    print("dataset:",data)

    print("variance of the dataset:",variance)
    print("standard deviation of the dataset:",standard_deviation)





if __name__=="__main__":
    main()
