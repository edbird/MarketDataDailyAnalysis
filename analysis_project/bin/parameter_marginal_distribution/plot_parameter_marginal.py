#!/usr/bin/env python3

import libmongo
from libexperiment.experiment_record_mongo import ExperimentRecord

import numpy
import matplotlib.pyplot as plt



def main():

    #data_mean = 0.116
    #data_stddev = 2.44

    x_success = None
    x_fail = None

    y_success = None
    y_fail = None

    try:
        x_success = numpy.load('mean_success.npy')
        x_fail = numpy.load('mean_fail.npy')

        y_success = numpy.load('stddev_success.npy')
        y_fail = numpy.load('stddev_fail.npy')

    except Exception as exception:
        print(f'failed to load data from disk, loading from MongoDB')

        connection = libmongo.get_connection_client()

        experiment_records_success = libmongo.get_experiment_records_optimize_success(connection, limit=85000)

        experiment_records_fail = libmongo.get_experiment_records_optimize_fail(connection, limit=85000)

        x_success = []
        y_success = []
        y_fail = []
        x_fail = []

        x_all = []
        y_all = []

        # add to data for success results
        for experiment_record in experiment_records_success:

            mean = experiment_record.get_optimize_mean()
            stddev = experiment_record.get_optimize_stddev()

            x_all.append(mean)
            y_all.append(stddev)

            x_success.append(mean)
            y_success.append(stddev)

        # add to data for failed results
        for experiment_record in experiment_records_fail:

            mean = experiment_record.get_optimize_mean()
            stddev = experiment_record.get_optimize_stddev()

            x_all.append(mean)
            y_all.append(stddev)

            x_fail.append(mean)
            y_fail.append(stddev)


        # convert to numpy array
        x_success = numpy.array(x_success)
        x_fail = numpy.array(x_fail)

        y_success = numpy.array(y_success)
        y_fail = numpy.array(y_fail)

        numpy.save('mean_success.npy', x_success)
        numpy.save('mean_fail.npy', x_fail)

        numpy.save('stddev_success.npy', y_success)
        numpy.save('stddev_fail.npy', y_fail)


    # plot 2x figures

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Parameter $\\mu$')
    axis.set_ylabel('Number of pseudo experiments')

    (bin_counts, bin_edges, _)      = axis.hist(x_success, bins=100, histtype='step', color='tab:blue')
    (bin_counts_2, bin_edges_2, _)  = axis.hist(x_fail, bins=100, histtype='step', color='tab:orange', range=(bin_edges[0], bin_edges[-1]))
    #axis.scatter(x_fail, y_fail, c='tab:purple', alpha=1.0, s=3)
    #axis.scatter(data_mean, data_stddev, c='white', alpha=1.0, s=10)

    figure.savefig('parameter_marginal_mean.png')
    figure.savefig('parameter_marginal_mean.pdf')

    #x_success = numpy.array(x_success)
    #x_fail = numpy.array(x_fail)
    #y_success = numpy.array(y_success)
    #y_fail = numpy.array(y_fail)

    #x_all = numpy.array(x_all)
    #y_all = numpy.array(y_all)

    #print(f'Success mean: {x_success.mean()} +- {x_success.std()}')
    #print(f'Fail mean: {x_fail.mean()} +- {x_fail.std()}')
    #print(f'Success stddev: {y_success.mean()} +- {y_success.std()}')
    #print(f'Fail stddev: {y_fail.mean()} +- {y_fail.std()}')
    #print(f'')
    #print(f'All mean: {x_all.mean()} +- {x_all.std()}')
    #print(f'All stddev: {y_all.mean()} +- {y_all.std()}')


    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Parameter $\\sigma$')
    axis.set_ylabel('Number of pseudo experiments')

    (bin_counts, bin_edges, _)      = axis.hist(y_success, bins=100, histtype='step', color='tab:blue')
    (bin_counts_2, bin_edges_2, _)  = axis.hist(y_fail, bins=100, histtype='step', color='tab:orange', range=(bin_edges[0], bin_edges[-1]))
    #axis.scatter(x_fail, y_fail, c='tab:purple', alpha=1.0, s=3)
    #axis.scatter(data_mean, data_stddev, c='white', alpha=1.0, s=10)

    figure.savefig('parameter_marginal_stddev.png')
    figure.savefig('parameter_marginal_stddev.pdf')


if __name__ == '__main__':
    main()
