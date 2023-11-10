#!/usr/bin/env python3


import math
import numpy
import matplotlib
import matplotlib.pyplot as plt
import datetime
from datetime import timezone
import scipy.stats as stats
import copy


def run_bootstrap(experimental_data, number_of_bins, histogram_range):

    number_of_experiments = len(experimental_data)

    (bin_counts, bin_edges) = numpy.histogram(experimental_data, bins=number_of_bins, range=histogram_range)

    print(f'run_bootstrap:')
    print(f'')
    print(f'experimental_data={experimental_data}')
    print(f'')
    print(f'bin_counts={bin_counts}')
    print(f'bin_edges={bin_edges}')

    bin_95_cl_low = numpy.zeros(bin_counts.shape)
    bin_95_cl_high = numpy.zeros(bin_counts.shape)

    bootstrap_data = numpy.zeros(shape=(number_of_experiments, number_of_bins), dtype=int)

    bootstrap_count = number_of_experiments # 100000 # should really be: number_of_experiments

    for count in range(bootstrap_count):

        # generate random integers to create a bootstrap dataset
        # the number of experiments included in each bootstrap is the total number of experiments
        # the full range of experimental data is used
        # hence why the argument number_of_experiments is used twice, we want this many random
        # integers, and the range of values is [0, number_of_experiments - 1]
        random_index = numpy.random.randint(number_of_experiments, size=number_of_experiments)
        #print(f'random_index: {random_index}')

        bootstrap_experimental_data = experimental_data[random_index]
        #print(f'bootstrap_experimental_data: {bootstrap_experimental_data}\n')

        # re-histogram the bootstrap dataset
        (bin_counts, bin_edges) = numpy.histogram(bootstrap_experimental_data, bins=number_of_bins, range=histogram_range)

        # save data
        bootstrap_data[count] = bin_counts

        if count == 0:
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)
            (bin_counts, bin_edges, _) = ax.hist(bootstrap_experimental_data, bins=number_of_bins, range=histogram_range)
            figure.savefig('bootstrap_experimental_data_0.png')


    #print(f'bootstrap_data:\n{bootstrap_data}')

    # find confidence levels
    for bin_index in range(number_of_bins):

        bootstrap_bin_data = copy.deepcopy(bootstrap_data[:, bin_index])

        sorted_bin_data = numpy.sort(bootstrap_bin_data)

        # check sorted
        if bin_index == 0:
            for element in sorted_bin_data:
                print(element)

        # find cl
        count_95_cl_low = float(bootstrap_count) * 0.5 * (1.0 - 0.95)
        count_95_cl_high = float(bootstrap_count) * 0.5 * (1.0 - 0.95)

        count_95_cl_low = round(count_95_cl_low)
        count_95_cl_high = round(count_95_cl_high)

        if bin_index == 0:
            print(f'count_95_cl_low={count_95_cl_low}')
            print(f'count_95_cl_high={count_95_cl_high}')

        # TODO: do some more debugging here to figure out if the right values are being calculated

        bin_95_cl_low[bin_index] = sorted_bin_data[count_95_cl_low]
        bin_95_cl_high[bin_index] = sorted_bin_data[-count_95_cl_high]

    return (bin_counts, bin_95_cl_low, bin_95_cl_high, bin_edges)


def generate_experiment_run_data(number_of_events, maximum_value):

    data = numpy.random.randint(maximum_value, size=number_of_events)

    #for data in data:
    #    if data >= 10:
    #        print(data)

    return data


def main():

    time_start = datetime.datetime.now(timezone.utc)

    maximum_value = 10 # there are 10 possible values in each experimental run
                        # bumped up to 30, because in theory all events could end up in the same bin, although this is extremely unlikely
    number_of_experimental_runs = 100000 #100000
    all_run_data = numpy.zeros((number_of_experimental_runs, maximum_value))
    all_run_data_index = 0

    # number of events generated per run
    number_of_events = 30

    while True:

        if all_run_data_index == number_of_experimental_runs:
            break

        # this is 1 run
        maximum_value = maximum_value # maximum value is not included, range is [0, 9] which is 10 possible values
        run_data = generate_experiment_run_data(number_of_events, maximum_value)

        #if all_run_data_index < 5:
        #    print(run_data)

        # number of bins for uniform distribution data, (a per experiment basis)
        number_of_bins = 10
        (bin_counts, bin_edges) = numpy.histogram(run_data, bins=number_of_bins, range=(0, maximum_value))

        #figure = plt.figure()
        #ax = figure.add_subplot(1, 1, 1)
        #(bin_counts, bin_edges, _) = ax.hist(run_data, bins=number_of_bins, range=(0, maximum_value))
        #figure.savefig('histogram.png')

        #print(f'bin_counts={bin_counts}, bin_edges={bin_edges}')
        # end of 1 run

        #if all_run_data_index < 5:
        #    print(f'{bin_counts}')

        all_run_data[all_run_data_index] = bin_counts

        all_run_data_index += 1

        plt.close()

    # draw Poisson distributed thing
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    # how many events are in bin `4` of the uniform data histogram, for each experimental run?
    print(f'all_run_data[4]={all_run_data[:, 4]}')
    poisson_data = all_run_data[:, 4]

    (bin_counts, bin_95_cl_low, bin_95_cl_high, bin_edges) = \
        run_bootstrap(poisson_data, number_of_bins=number_of_events, histogram_range=(0, number_of_events))

    print('CL low')
    print(bin_95_cl_low)
    print('central value')
    print(bin_counts)
    print('CL high')
    print(bin_95_cl_high)
    print('edges')
    print(bin_edges)

    ybot = bin_counts - bin_95_cl_low
    ytop = bin_95_cl_high - bin_counts

    for index, value in enumerate(ybot):
        if value < 0.0:
            print(f'ybot, negative value: index={index}, bin_count={bin_counts[index]}, bin_95_cl_low={bin_95_cl_low[index]}')
            ybot[index] = 0.0

    for index, value in enumerate(ytop):
        if value < 0.0:
            print(f'ytop, negative value: index={index}, bin_count={bin_counts[index]}, bin_95_cl_high={bin_95_cl_high[index]}')
            ytop[index] = 0.0

    ax.errorbar(x=bin_edges[:-1], y=bin_counts, yerr=(ybot, ytop), linestyle='none', marker='_')
    #(bin_counts, bin_edges, _) = ax.hist(poisson_data, bins=number_of_events, range=(0, number_of_events))
    # TODO:


    # Poisson model
    lambda_param = float(number_of_events) / float(maximum_value)

    x_model = numpy.linspace(0, number_of_events, num=(number_of_events + 1), dtype=int)
    y_model = numpy.zeros(number_of_events + 1)

    #index = 0
    #for x in x_model:
    #    amplitude = number_of_experimental_runs
    #    y_model[index] = amplitude * numpy.power(lambda_param, x) / math.factorial(x) * numpy.exp(-lambda_param)
    #    index += 1

    #y_model = numpy.power(lambda_param, x_model) / numpy.math.factorial(x_model) * numpy.exp(-lambda_param)
    amplitude = number_of_experimental_runs
    y_model = amplitude * stats.poisson(lambda_param).pmf(x_model)

    ax.scatter(x_model, y_model, color='red')
    #ax.set_yscale('log')
    #ax.set_ylim(ymin=0.1, ymax=None)

    figure.savefig('poisson_histogram_0.png')

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':

    main()

