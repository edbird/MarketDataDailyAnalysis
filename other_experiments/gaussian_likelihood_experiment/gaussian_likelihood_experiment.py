#!/usr/bin/env python3


import math
import numpy
import matplotlib
import matplotlib.pyplot as plt
import datetime
from datetime import timezone
import scipy
import copy




def run_1_experiment(batch_size, mean, stddev, debug=False):

    mean_model = 0.0
    stddev_model = 1.0

    likelihood_value = 1.0

    random_gaussian_values = numpy.random.normal(mean, stddev, size=batch_size) # TODO: move to new style RNG

    normalization_factor = 0.5 * 1.0 / math.sqrt(math.pi)
    likelihood_values = scipy.stats.norm.pdf(random_gaussian_values, mean_model, stddev_model) / normalization_factor
    log_likelihood_values = numpy.log(likelihood_values)

    likelihood_value = likelihood_values.prod()
    log_likelihood_value = log_likelihood_values.sum()

    if debug:
        print(f'gaussian values: {random_gaussian_values}')
        print(f'likelihood values: {likelihood_values}')
        print(f'likelihood: {likelihood_value}')

    return likelihood_value


def run_experiment(mean, stddev):

    batch_size = 3

    number_of_experiments = 100000

    likelihood_value_per_experiment = numpy.zeros(number_of_experiments)

    for index in range(number_of_experiments):
        likelihood_value_per_experiment[index] = run_1_experiment(batch_size, mean, stddev, index == 0)

    return likelihood_value_per_experiment


def plot_results(*likelihood_values_per_experiment, bins=100):

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    for experiment in likelihood_values_per_experiment:
        ax.hist(experiment, bins, alpha=0.2)

    figure.savefig('gaussian-likelihood-experiment.pdf')


def plot_single_experiment(likelihood_values, cl_value, bins=100):

    figure = plt.figure()
    p1 = ax = figure.add_subplot(1, 1, 1)

    (bin_counts, bin_edges, _) = ax.hist(likelihood_values, bins, histtype='stepfilled', alpha=0.5)
    likelihood_value_observed = likelihood_values[0]
    x = likelihood_value_observed
    x_index = numpy.digitize(x, bin_edges)
    y = bin_counts[x_index]
    print(f'x={x}, y={y}')
    p2 = ax.plot([x, x], [0.0, y], 'k-')

    ax.set_xlabel('Likelihood Value')
    ax.set_ylabel('Number of Experiments')

    s = f'CL = {round(cl_value * 100.0, ndigits=2)} %'
    ax.text(x, y * 1.2, s)

    figure.savefig('gaussian-likelihood-experiment-single.pdf')


def main():

    time_start = datetime.datetime.now(timezone.utc)

    likelihood_values_per_experiment_0 = run_experiment(0.0, 1.0)
    likelihood_values_per_experiment_1 = run_experiment(1.0, 1.0)
    likelihood_values_per_experiment_2 = run_experiment(0.0, 2.0)
    likelihood_values_per_experiment_3 = run_experiment(0.0, 0.5)

    plot_results(
        likelihood_values_per_experiment_0, \
        likelihood_values_per_experiment_1, \
        likelihood_values_per_experiment_2, \
        likelihood_values_per_experiment_3)

    hist_0 = numpy.histogram(likelihood_values_per_experiment_0, bins=100)
    (bin_counts, bin_edges) = hist_0
    bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print(f'shape: {(bin_midpoints * bin_counts).shape}')
    mean = (bin_midpoints * bin_counts).sum() / bin_counts.sum()
    print(f'mean={mean}')

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')

    # Use the results generated in Experiment 0 to calculate the 95 % CL

    # The observed value (pretend this one comes from some observation)
    # Could more sensibly perhaps calculate the mid value (median)
    #likelihood_value_observed = likelihood_values_per_experiment_0[0]
    #likelihood_distribution_values = likelihood_values_per_experiment_0[1:]
    #likelihood_distribution_values_sorted = likelihood_distribution_values.sort()
    #number_of_values = len(likelihood_distribution_values_sorted)
    #cl_low_index = round(5.0e-2 * float(number_of_values))
    ##cl_high_index = round((1.0 - 2.5e-2) * float(number_of_values))
    ##print(f'index CL low, high: {cl_low_index}, {cl_high_index}')
    #print(f'index CL: {cl_low_index}')

    # Use the results generated in Experiment 0 to calculate the CL value

    # The observed value (pretend this one comes from some observation)
    likelihood_value_observed = likelihood_values_per_experiment_0[0]
    # Other values
    likelihood_distribution_values = likelihood_values_per_experiment_0[1:]

    count = 0
    for value in likelihood_distribution_values:
        if value > likelihood_value_observed:
            count += 1

    # P(ll > ll_obs) = ...
    cl_value = float(count) / float(len(likelihood_values_per_experiment_0))
    print(f'count={count}, len={len(likelihood_values_per_experiment_0)}, CL={cl_value}')

    plot_single_experiment(likelihood_values_per_experiment_0, cl_value)


if __name__ == '__main__':
    main()

