#!/usr/bin/env python3

import libmongo

from libplot import plot_failed_fit
from libexperiment.pseudodata_experiment import retry_run_pseudodata_experiment_get_optimize_result
from libexperiment.experiment_record_mongo import ExperimentRecord

import numpy


def driver():

    connection = libmongo.get_connection_client()

    experiment_records = libmongo.get_experiment_records_optimize_fail(connection, limit=None)

    print(f'obtained {len(experiment_records)} failed optimization experiments')

    for experiment_record in experiment_records:

        process_experiment_record(experiment_record)


def process_experiment_record(experiment_record):

    experiment_id = experiment_record.get_experiment_id()
    pseudodata = experiment_record.get_dataset()

    # original optimization parameters
    optimize_mean = experiment_record.get_optimize_mean()
    optimize_stddev = experiment_record.get_optimize_stddev()

    # convert to numpy
    pseudodata = numpy.array(pseudodata)

    # re-perform the fit with the original method to find out what the message was
    (experiment_record_2, optimize_result) = \
        retry_run_pseudodata_experiment_get_optimize_result(pseudodata, experiment_id, 'default')

    # values should be the same
    optimize_mean_2 = experiment_record_2.get_optimize_mean()
    optimize_stddev_2 = experiment_record_2.get_optimize_stddev()

    if check_numerical_difference(optimize_mean, optimize_mean_2, 1.0e-6):
        print(f'{optimize_mean}, {optimize_mean_2}')

    if check_numerical_difference(optimize_stddev, optimize_stddev_2, 1.0e-6):
        print(f'{optimize_stddev}, {optimize_stddev_2}')


    if optimize_result.success != True:
        message = optimize_result.message

        if message != 'Desired error not necessarily achieved due to precision loss.':
            print(f'found serious fit conversion failure, message={message}')
            print('optimize result')
            print(f'{optimize_result}')
            print(f'message={message}')
    elif optimize_result.success == True:
        print(f'error: somehow the fix managed to produce a successful result this time')
        print(f'this should never happen')


    # re-perform the fit with simplex method to check
    (experiment_record_3, optimize_result) = \
        retry_run_pseudodata_experiment_get_optimize_result(pseudodata, experiment_id, 'simplex')

    # check success
    if optimize_result.success != True:
        message = optimize_result.message

        print(f'fit failed with SIMPLEX method, this is a serious failure')
        print(f'message={message}')

    # check values compatiable
    optimize_mean_3 = experiment_record_3.get_optimize_mean()
    optimize_stddev_3 = experiment_record_3.get_optimize_stddev()

    if check_numerical_difference(optimize_mean, optimize_mean_3, 1.0e-3):
        print(f'{optimize_mean}, {optimize_mean_3}')

    if check_numerical_difference(optimize_stddev, optimize_stddev_3, 1.0e-3):
        print(f'{optimize_stddev}, {optimize_stddev_3}')


    #(bin_contents, bin_edges) = numpy.histogram(pseudodata, bins=20)

    # Create values of bin midpoints
    #bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    #plot_failed_fit(experiment_id, bin_midpoints, bin_contents, optimize_mean, optimize_stddev)


def check_numerical_difference(x, y, percentage_threshold):

    difference = x - y
    abs_difference = abs(difference)
    abs_value = abs(x)
    fraction = abs_difference / abs_value
    percentage = fraction * 100.0

    if percentage >= percentage_threshold:
        return True
    else:
        return False


def test_check_numerical_difference():

    x = 1000.0
    y = 1001.0

    if check_numerical_difference(x, y, 1.0):
        raise "check_numerical_difference 1"

    if check_numerical_difference(x, y, 0.01) == False:
        raise "check_numerical_difference 2"


def main():
    test_check_numerical_difference()
    driver()


if __name__ == '__main__':
    main()
