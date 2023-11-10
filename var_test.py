#!/usr/bin/env python3


import pandas
import math
import numpy


def main():

    #data_list = [1.0, 2.0, 3.0, 4.0, 10.0]
    data_list = [3.0, 4.0]

    df = pandas.DataFrame(data_list, columns=['Example'])

    df_var = df['Example'].var()
    print(f'df_var={df_var}')

    sum = 0.0
    sum2 = 0.0
    count = 0
    for x in df['Example']:

        count += 1
        sum += x
        sum2 += x * x

    mean = sum / float(count)
    mean2 = sum2 / float(count - 1)
    #var = mean2 - (float(count) / float(count - 1)) * mean * mean
    #var = (sum2) / float(count - 1)  - (float(count) / float(count - 1)) * mean * mean
    var = (sum2) / float(count - 1)  - (1.0 / float(count) / float(count - 1)) * sum * sum
    print(f'var={var}')



if __name__ == '__main__':
    main()

