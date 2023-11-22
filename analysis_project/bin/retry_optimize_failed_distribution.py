#!/usr/bin/env python3

import libmongo

from libplot import plot_failed_fit
from libexperiment.pseudodata_experiment import retry_run_pseudodata_experiment_get_optimize_result

import numpy


def driver(skip=None):

    connection = libmongo.get_connection_client()

    experiment_record = libmongo.get_experiment_record_optimize_fail(connection, skip)

    experiment_id = experiment_record.get_experiment_id()
    pseudodata = experiment_record.get_dataset()

    # convert to numpy
    pseudodata = numpy.array(pseudodata)

    (experiment_record_2, optimize_result) = \
        retry_run_pseudodata_experiment_get_optimize_result(pseudodata, experiment_id)

    optimize_mean_2 = experiment_record_2.get_optimize_mean()
    optimize_stddev_2 = experiment_record_2.get_optimize_stddev()

    optimize_mean = experiment_record.get_optimize_mean()
    optimize_stddev = experiment_record.get_optimize_stddev()

    print(f'{optimize_mean}, {optimize_mean_2}')
    print(f'{optimize_stddev}, {optimize_stddev_2}')

    if optimize_result.success != True:
        print('optimize result')
        print(f'{optimize_result}')
        print(f'message={optimize_result.message}')

    (bin_contents, bin_edges) = numpy.histogram(pseudodata, bins=20)

    # Create values of bin midpoints
    bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plot_failed_fit(experiment_id, bin_midpoints, bin_contents, optimize_mean, optimize_stddev)


def main():

    #for skip in range(10):
    #    driver(skip)

    driver()


if __name__ == '__main__':
    main()
