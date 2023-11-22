#!/usr/bin/env python3

import libmongo

from libplot import plot_failed_fit

import numpy


def driver(skip):

    connection = libmongo.get_connection_client()

    experiment_record = libmongo.get_experiment_record_optimize_fail(connection, skip)

    experiment_id = experiment_record.get_experiment_id()
    pseudodata = experiment_record.get_dataset()
    optimize_mean = experiment_record.get_optimize_mean()
    optimize_stddev = experiment_record.get_optimize_stddev()

    (bin_contents, bin_edges) = numpy.histogram(pseudodata, bins=20)

    # Create values of bin midpoints
    bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plot_failed_fit(experiment_id, bin_midpoints, bin_contents, optimize_mean, optimize_stddev)


def main():

    for skip in range(10):
        driver(skip)


if __name__ == '__main__':
    main()
