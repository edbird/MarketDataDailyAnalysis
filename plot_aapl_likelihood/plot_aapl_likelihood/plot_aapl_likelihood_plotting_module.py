

import numpy
import scipy.stats as stats
import matplotlib.pyplot as plt


def plot_diff_with_stddev(eod_aapl_us_diff, aapl_us_diff_std):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change')
    ax.grid(True, axis='both', alpha=0.3, which='both')

    (bin_contents, bin_edges, _) = \
        ax.hist(eod_aapl_us_diff, label='AAPL', bins=20, density=False)

    #curve_data_x = numpy.linspace(-5 * aapl_us_diff_std, 5 * aapl_us_diff_std, 1000)
    curve_data_x = numpy.linspace(bin_edges[0], bin_edges[-1], 1000)
    normalization_constant = bin_contents.sum()
    curve_data_y = normalization_constant * stats.norm.pdf(curve_data_x, 0.0, aapl_us_diff_std)

    ax.plot(curve_data_x, curve_data_y)

    fig.savefig('eod_aapl_us_diff.png')


def plot_failed_fit(index, bin_midpoints, bin_contents, amplitude, mean, stddev):
    print(f'plotting failed fit, index={index}')

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

    fig.savefig(f'failed_fit_figure/failed_fit_{index}.png')
    plt.close()


def plot_bootstrap_data(iteration, bootstrap_sample):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change, Bootstrap Data')
    ax.grid(True, axis='both', alpha=0.3, which='both')

    (bin_contents, bin_edges, _) = \
        ax.hist(bootstrap_sample, label='AAPL', bins=20, density=False)

    fig.savefig(f'bootstrap_data_figure/bootstrap_{iteration}.png')
    plt.close()
