#!/usr/bin/env python3

import libmongo
from libexperiment.experiment_record_mongo import ExperimentRecord

import numpy
import matplotlib.pyplot as plt



def main():

    #data_mean = 0.116
    #data_stddev = 2.44

    x_all = []
    y_all = []

    try:
        x_all = numpy.load('mean.npy')
        y_all = numpy.load('stddev.npy')

    except Exception as exception:
        print(f'failed to load data from disk, loading from MongoDB')

        connection = libmongo.get_connection_client()

        experiment_records = libmongo.get_experiment_records(connection)

        # add to data for success results
        for experiment_record in experiment_records:

            mean = experiment_record.get_optimize_mean()
            stddev = experiment_record.get_optimize_stddev()

            x_all.append(mean)
            y_all.append(stddev)

        # convert to numpy array
        x_all = numpy.array(x_all)
        y_all = numpy.array(y_all)

        numpy.save('mean.npy', x_all)
        numpy.save('stddev.npy', y_all)

    # plot 2x figures

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Parameter $\\mu$')
    axis.set_ylabel('Number of pseudo experiments')
    axis.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    (bin_counts, bin_edges, _)      = axis.hist(x_all, bins=100, histtype='step', color='tab:blue')
    #axis.scatter(x_fail, y_fail, c='tab:purple', alpha=1.0, s=3)
    #axis.scatter(data_mean, data_stddev, c='white', alpha=1.0, s=10)

    figure.savefig('parameter_marginal_mean_all.png')
    figure.savefig('parameter_marginal_mean_all.pdf')

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
    axis.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    (bin_counts, bin_edges, _)      = axis.hist(y_all, bins=100, histtype='step', color='tab:blue')
    #axis.scatter(x_fail, y_fail, c='tab:purple', alpha=1.0, s=3)
    #axis.scatter(data_mean, data_stddev, c='white', alpha=1.0, s=10)

    figure.savefig('parameter_marginal_stddev_all.png')
    figure.savefig('parameter_marginal_stddev_all.pdf')
    
    print(f'Parameter mean:')
    x_mean = x_all.mean()
    print(f'{x_mean}')
    print(f'{numpy.percentile(x_all, [2.5, 50, 97.5])}')
    x_low = numpy.percentile(x_all, 2.5)
    x_high = numpy.percentile(x_all, 97.5)
    print(f'{x_mean}+{x_high - x_mean}-{x_mean - x_low}')
    
    print(f'Parameter stddev')
    y_mean = y_all.mean()
    print(f'{y_mean}')
    print(f'{numpy.percentile(y_all, [2.5, 50, 97.5])}')
    y_low = numpy.percentile(y_all, 2.5)
    y_high = numpy.percentile(y_all, 97.5)
    print(f'{y_mean}+{y_high - y_mean}-{y_mean - y_low}')


if __name__ == '__main__':
    main()
