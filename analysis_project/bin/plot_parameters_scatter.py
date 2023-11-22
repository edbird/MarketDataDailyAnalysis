#!/usr/bin/env python3

import libmongo
from libexperiment.experiment_record_mongo import ExperimentRecord

import numpy
import matplotlib.pyplot as plt



def main():

    connection = libmongo.get_connection_client()

    experiment_records = libmongo.get_experiment_records(connection, limit=20000)

    x_success = []
    y_success = []
    y_fail = []
    x_fail = []

    for experiment_record in experiment_records:

        mean = experiment_record.get_optimize_mean()
        stddev = experiment_record.get_optimize_stddev()

        if experiment_record.get_optimize_success() == True:
            x_success.append(mean)
            y_success.append(stddev)

        elif experiment_record.get_optimize_success() == False:
            x_fail.append(mean)
            y_fail.append(stddev)


    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Parameter $\\mu$')
    axis.set_ylabel('Parameter $\\sigma$')

    axis.scatter(x_success, y_success, c='tab:orange', alpha=0.1, s=3)
    axis.scatter(x_fail, y_fail, c='tab:purple', alpha=1.0, s=3)

    figure.savefig('parameter_scatter.png')

    x_success = numpy.array(x_success)
    x_fail = numpy.array(x_fail)
    y_success = numpy.array(y_success)
    y_fail = numpy.array(y_fail)

    print(f'Success mean: {x_success.mean()} +- {x_success.std()}')
    print(f'Fail mean: {x_fail.mean()} +- {x_fail.std()}')
    print(f'Success stddev: {y_success.mean()} +- {y_success.std()}')
    print(f'Fail stddev: {y_fail.mean()} +- {y_fail.std()}')


if __name__ == '__main__':
    main()
