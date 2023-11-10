#!/usr/bin/env python3


import matplotlib
import matplotlib.pyplot as plt

import pandas
import math

import numpy
import scipy
import scipy.stats as stats


def main():

    # Load AAPL
    eod_aapl_us = pandas.read_csv('eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    # Load NVDA
    eod_nvda_us = pandas.read_csv('eod_nvda_us.csv', dtype='str', delimiter=',')
    eod_nvda_us['Date'] = pandas.to_datetime(eod_nvda_us['Date'])
    eod_nvda_us['Close'] = eod_nvda_us['Close'].apply(lambda close: float(close))


    aapl_us_var = eod_aapl_us['Close'].var()
    nvda_us_var = eod_nvda_us['Close'].var()
    aapl_us_std = eod_aapl_us['Close'].std()
    nvda_us_std = eod_nvda_us['Close'].std()
    aapl_us_mean = eod_aapl_us['Close'].mean()
    nvda_us_mean = eod_nvda_us['Close'].mean()

    print(f'Pandas Calculations for Var, Std, Mean:')
    print(f'Var: AAPL={aapl_us_var}, NVDA={nvda_us_var}')
    print(f'Std: AAPL={aapl_us_std}, NVDA={nvda_us_std}')
    print(f'Mean: AAPL={aapl_us_mean}, NVDA={nvda_us_mean}')

    sum = 0.0
    sum2 = 0.0
    count = 0
    for close in eod_aapl_us['Close']:
        count += 1
        sum += close
        sum2 += close * close

    #print(f'sum2={sum2}, sum={sum}, sum*sum={sum*sum}')
    mean = sum / float(count)
    var = (sum2) / float(count - 1)  - (float(count) / float(count - 1)) * mean * mean
    var = (sum2) / float(count - 1)  - (1.0 / float(count) / float(count - 1)) * sum * sum

    print(f'Calculation of Var obtained from sum and sum**2:')
    print(f'N={count}, var={var}, sum2={sum2}, mean={mean}, mean^2={mean*mean}')


    # alternative calculation
    # slightly different result but compatiable
    var_2 = 0.0
    for close in eod_aapl_us['Close']:

        x = abs(close - mean) ** 2.0
        var_2 += x

    var_2 = var_2 / float(count - 1)
    print(f'Calculation of Var obtained from Sum[abs(x - mean)**2]:')
    print(f'var_2={var_2}')

    print(f'All the above values for Var should agree')


if __name__ == '__main__':
    main()
