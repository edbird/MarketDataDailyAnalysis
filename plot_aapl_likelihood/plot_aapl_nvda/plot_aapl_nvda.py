#!/usr/bin/env python3


import matplotlib
import matplotlib.pyplot as plt

import pandas
import math

import numpy
import scipy
import scipy.stats as stats


def main():

    """
        Plot AAPL and NVDA stock price across time
    """

    # Load AAPL
    eod_aapl_us = pandas.read_csv('./market_data/eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    # Load NVDA
    eod_nvda_us = pandas.read_csv('./market_data/eod_nvda_us.csv', dtype='str', delimiter=',')
    eod_nvda_us['Date'] = pandas.to_datetime(eod_nvda_us['Date'])
    eod_nvda_us['Close'] = eod_nvda_us['Close'].apply(lambda close: float(close))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    #fig.autofmt_xdate(rotation=90)
    ax.grid(True, axis='both', alpha=0.3, which='both')

    ax.scatter(eod_aapl_us['Date'], eod_aapl_us['Close'], label='AAPL')
    ax.scatter(eod_nvda_us['Date'], eod_nvda_us['Close'], label='NVDA')
    plt.legend(loc='upper left')
    fig.savefig('eod_aapl_nvda_us.png')


if __name__ == '__main__':
    main()
