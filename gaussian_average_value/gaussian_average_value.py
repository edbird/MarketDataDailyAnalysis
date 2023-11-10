#!/usr/bin/env python3


import math
import numpy
import matplotlib
import matplotlib.pyplot as plt
import datetime
from datetime import timezone
import scipy
import copy


def run_experiment():

    mean = 0.0
    stddev = 1.0

    batch_size = 100000
    random_gaussian_values = numpy.random.normal(mean, stddev, size=batch_size)

    random_gaussian_values = scipy.stats.norm.pdf(random_gaussian_values, mean, stddev)

    average = random_gaussian_values.sum() / float(batch_size)

    return average


def plot_results(average_gaussian_values, bins=100):

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    ax.hist(average_gaussian_values, bins)

    figure.savefig('gaussian-average-value.pdf')


def main():

    time_start = datetime.datetime.now(timezone.utc)

    number_of_experiments = 10000
    average_gaussian_values = numpy.zeros(number_of_experiments)

    for index in range(number_of_experiments):
        average_gaussian_values[index] = run_experiment()

    print(f'average gaussian: {average_gaussian_values.mean()} +- {average_gaussian_values.std()}')
    print(f'Integral of Gaussian squared: {0.5 * 1.0 / math.sqrt(math.pi)}')

    plot_results(average_gaussian_values)

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':

    main()


