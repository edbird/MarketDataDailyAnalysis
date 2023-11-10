#!/usr/bin/env python3


import matplotlib
import matplotlib.pyplot as plt

import pandas
import math

import numpy
import scipy
import scipy.stats as stats


def gaussian(x, amplitude, mean, stddev):

    y = numpy.power((x - mean) / stddev, 2.0)
    return amplitude * numpy.exp(-0.5 * y)


def main():

    # Load AAPL
    eod_aapl_us = pandas.read_csv('./market_data/eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    # Load NVDA
    eod_nvda_us = pandas.read_csv('./market_data/eod_nvda_us.csv', dtype='str', delimiter=',')
    eod_nvda_us['Date'] = pandas.to_datetime(eod_nvda_us['Date'])
    eod_nvda_us['Close'] = eod_nvda_us['Close'].apply(lambda close: float(close))

    # Create difference timeseries
    eod_aapl_us['Diff'] = eod_aapl_us['Close'].diff()
    aapl_us_diff_std = eod_aapl_us['Diff'].std()

    # plot histogram of difference data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change')
    #fig.autofmt_xdate(rotation=90)
    ax.grid(True, axis='both', alpha=0.3, which='both')

    (n, bins, _) = ax.hist(eod_aapl_us['Diff'], label='AAPL', bins=20, density=False)
    curve_data_x = numpy.linspace(-5 * aapl_us_diff_std, 5 * aapl_us_diff_std, 1000)
    normalization_constant = n.sum()
    #normalization_constant_2 = 1.0 / (math.sqrt(2.0 * math.pi) * aapl_us_diff_std)
    curve_data_y = normalization_constant * stats.norm.pdf(curve_data_x, 0.0, aapl_us_diff_std)
    #curve_data_y = gaussian(curve_data_x, normalization_constant / math.sqrt(2 * math.pi * aapl_us_diff_std * aapl_us_diff_std), 0.0, aapl_us_diff_std)
    ax.plot(curve_data_x, curve_data_y)
    fig.savefig('eod_aapl_us_diff_.png')

    print(f'bins:\n{bins}')

    # Create values of bin midpoints
    bin_midpoints = 0.5 * (bins[1:] + bins[:-1])

    hist_data = n
    #hist_model = 250.0 / 236.0532057 * normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)
    hist_model = normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)
    ax.plot(bin_midpoints, hist_model)
    fig.savefig('eod_aapl_us_diff_2_.png')

    print(f'normalization_constant={normalization_constant}')
    #print(f'normalization_constant_2={normalization_constant_2}')
    print(f'{len(hist_data), len(hist_model)}')
    print(hist_data)
    print(hist_model)



if __name__ == '__main__':
    main()
