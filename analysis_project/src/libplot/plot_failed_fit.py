
import math
import numpy
import matplotlib.pyplot as plt

import scipy.stats as stats


def plot_failed_fit(experiment_id, bin_midpoints, bin_contents, mean, stddev):
    print(f'plotting failed fit, index={experiment_id}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change, Bootstrap Data')
    ax.grid(True, axis='both', alpha=0.3, which='both')

    ax.step(bin_midpoints, bin_contents)
    x = numpy.linspace(bin_midpoints[0], bin_midpoints[-1], 1000)
    bin_width = (bin_midpoints[-1] - bin_midpoints[0]) / len(bin_midpoints)
    area = bin_contents.sum() * bin_width
    print(f'area scaling factor for amplitude: {area}')
    #/ (math.sqrt(2.0 * math.pi) * stddev)
    amplitude = 1.0 * area
    y = amplitude * stats.norm.pdf(x, mean, stddev)
    ax.plot(x, y, 'm-')
    ymax = 1.2 * bin_contents.max()
    ax.set_ylim(ymax=ymax)

    fig.savefig(f'failed_fit_figure/failed_fit_{experiment_id}.png')
    plt.close()


def plot_failed_fit_3param(experiment_id, bin_midpoints, bin_contents, amplitude, mean, stddev):
    print(f'plotting failed fit, index={experiment_id}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change, Bootstrap Data')
    ax.grid(True, axis='both', alpha=0.3, which='both')

    ax.step(bin_midpoints, bin_contents)
    x = numpy.linspace(bin_midpoints[0], bin_midpoints[-1], 1000)
    y = amplitude * stats.norm.pdf(x, mean, stddev)
    ax.plot(x, y, 'm-')
    ymax = 1.2 * bin_contents.max()
    ax.set_ylim(ymax=ymax)

    fig.savefig(f'failed_fit_figure/failed_fit_{experiment_id}.png')
    plt.close()

