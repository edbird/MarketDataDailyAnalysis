#!/usr/bin/env python3

import libmongo

from libplot import plot_log_likelihood_distribution_both

def main():

    # load data from MongoDB

    connection = libmongo.get_connection_client()

    log_likelihood_values_success = \
        libmongo.get_log_likelihood_values_optimize_success(connection, limit=85000)

    log_likelihood_values_fail = \
        libmongo.get_log_likelihood_values_optimize_fail(connection, limit=85000)

    plot_log_likelihood_distribution_both(None, log_likelihood_values_success, log_likelihood_values_fail, filename='log_likelihood_distribution_both.png')
    plot_log_likelihood_distribution_both(None, log_likelihood_values_success, log_likelihood_values_fail, filename='log_likelihood_distribution_both.pdf')


if __name__ == '__main__':

    main()

