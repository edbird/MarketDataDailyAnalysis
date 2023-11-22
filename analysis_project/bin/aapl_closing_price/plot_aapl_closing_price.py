#!/usr/bin/env python3

from libloaddata import load_data_aapl

import matplotlib.pyplot as plt

import pandas


def main():

    """
        Plot AAPL and NVDA stock price across time
    """

    # load data
    eod_aapl_us = load_data_aapl()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('End of Day Closing Price')
    ax.grid(True, axis='both', alpha=0.3, which='both')

    ax.scatter(eod_aapl_us['Date'], eod_aapl_us['Close'], label='AAPL', s=5)
    plt.legend(loc='upper left')
    fig.savefig('aapl_eod_closing_price.png')
    fig.savefig('aapl_eod_closing_price.pdf')


if __name__ == '__main__':
    main()
