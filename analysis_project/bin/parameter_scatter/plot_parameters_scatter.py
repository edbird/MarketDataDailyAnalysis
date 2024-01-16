#!/usr/bin/env python3

import libmongo
from libexperiment.experiment_record_mongo import ExperimentRecord

import numpy
import matplotlib.pyplot as plt



def main():

    data_mean = 0.116
    data_stddev = 2.44

    connection = libmongo.get_connection_client()

    experiment_records = libmongo.get_experiment_records(connection, limit=20000)

    x_success = []
    y_success = []
    y_fail = []
    x_fail = []

    x_all = []
    y_all = []

    for experiment_record in experiment_records:

        mean = experiment_record.get_optimize_mean()
        stddev = experiment_record.get_optimize_stddev()

        x_all.append(mean)
        y_all.append(stddev)

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
    axis.scatter(data_mean, data_stddev, c='white', alpha=1.0, s=10)

    figure.savefig('parameter_scatter.png')
    figure.savefig('parameter_scatter.pdf')

    x_success = numpy.array(x_success)
    x_fail = numpy.array(x_fail)
    y_success = numpy.array(y_success)
    y_fail = numpy.array(y_fail)

    x_all = numpy.array(x_all)
    y_all = numpy.array(y_all)

    print(f'Success mean: {x_success.mean()} +- {x_success.std()}')
    print(f'Success stddev: {y_success.mean()} +- {y_success.std()}')
    
    if len(x_fail) > 0:
        print(f'Fail mean: {x_fail.mean()} +- {x_fail.std()}')
    if len(y_fail) > 0:
        print(f'Fail stddev: {y_fail.mean()} +- {y_fail.std()}')
        
    print(f'')
    print(f'All mean: {x_all.mean()} +- {x_all.std()}')
    print(f'All stddev: {y_all.mean()} +- {y_all.std()}')


if __name__ == '__main__':
    main()
