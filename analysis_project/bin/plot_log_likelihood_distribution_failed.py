#!/usr/bin/env python3

import libmongo

from libplot import plot_log_likelihood_distribution

def main():

    # load data from MongoDB

    connection = libmongo.get_connection_client()
    log_likelihood_values = \
        libmongo.get_log_likelihood_values_optimize_fail(connection, limit=10000)

    plot_log_likelihood_distribution(None, log_likelihood_values, filename='log_likelihood_distribution_failed.png')


if __name__ == '__main__':

    main()

