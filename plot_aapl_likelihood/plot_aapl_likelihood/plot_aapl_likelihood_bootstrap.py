#!/usr/bin/env python3


import matplotlib
import matplotlib.pyplot as plt

import pandas
import math

import numpy
import scipy
import scipy.stats as stats

from multiprocessing import Pool

import datetime
from datetime import timezone
import os


class OptimizeFunctionException(Exception):

    def __init__(self, amplitude, mean, stddev):
        self.amplitude = amplitude
        self.mean = mean
        self.stddev = stddev


def optimize_function_likelihood(p, x, y):

    amplitude = p[0]
    mean = p[1]
    stddev = p[2]

    #print('optimize:')
    #print(amplitude)
    #print(mean)
    #print(stddev)
    #print(x)
    #print(y)

    # the change in price is expected to follow an exponential
    # distribution with a mean which is close to but not quite
    # zero
    normalization_constant = amplitude
    model = normalization_constant * stats.norm.pdf(x, mean, stddev)

    #fval = 1.0
    fval = 0.0

    # to create the likelihood function, for each bin midpoint (x)
    # get the value of the PDF, this gives the expected number of
    # events in this bin. Assume that the number of events in the
    # bin is distributed according to a Poisson distribution. The
    # total PDF for the whole histogram is a set of Poissons
    # multiplied together

    for index in range(len(model)):
        poisson_value = stats.poisson(model[index]).pmf(y[index])
        if poisson_value <= 0.0:
            print(f'\nindex: {index}, model: {model[index]}, y: {y[index]}, poisson: {poisson_value}')
            print(f'A={normalization_constant}, mean={mean}, stddev={stddev}')

            raise OptimizeFunctionException(amplitude, mean, stddev)

        log_poisson_value = math.log(poisson_value)

        #fval *= poisson_value
        fval += log_poisson_value
        #print(f'fval={fval}')

    # minimize the negative, gives a maximization
    return -fval


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

    fig.savefig('eod_aapl_us_diff_2.png')


def perform_fitting_procedure(data, x0, index, disp=False):

    # Get histogram data
    (bin_contents, bin_edges) = \
        numpy.histogram(data, bins=20)

    # Create values of bin midpoints
    bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    try:
        optimize_result = scipy.optimize.minimize(optimize_function_likelihood, x0, method='BFGS', \
            args=(bin_midpoints, bin_contents), \
            options={'disp': disp})

        return optimize_result

    except OptimizeFunctionException as exception:

        amplitude = exception.amplitude
        mean = exception.mean
        stddev = exception.stddev
        plot_failed_fit(index, bin_midpoints, bin_contents, amplitude, mean, stddev)

    except Exception as exception:
        print(f'failed fitting procedure: index={index}')
        print(f'{exception}')

    return None


def save_ll_values(ll_values):

    filename = 'll_values.numpy'
    numpy.savetxt(filename, ll_values)


def load_ll_values(target_number_of_ll_values):

    ll_values = numpy.zeros(target_number_of_ll_values)

    number_of_ll_values_loaded = None
    ll_values_from_file = None
    filename = 'll_values.numpy'

    if os.path.exists(filename):
        ll_values_from_file = numpy.loadtxt(filename)
        number_of_ll_values_loaded = len(ll_values_from_file)
        print(f'number of values loaded: {number_of_ll_values_loaded}')

        for index, value in enumerate(ll_values_from_file):
            # excess values are ignored and will not be returned
            if index < len(ll_values):
                ll_values[index] = value

    if number_of_ll_values_loaded is None:
        number_of_ll_values_loaded = 0

    number_of_values = min(number_of_ll_values_loaded, len(ll_values))

    return (ll_values, number_of_values)


def generate_bootstrap_data(eod_aapl_us_diff, number_of_bootstrap_samples):

    bootstrap_data = numpy.zeros(shape=(number_of_bootstrap_samples, len(eod_aapl_us_diff)), dtype=float)

    number_of_rows_loaded = None
    bootstrap_data_from_file = None
    filename = 'bootstrap_data.numpy'

    if os.path.exists(filename):
        bootstrap_data_from_file = numpy.loadtxt(filename)
        number_of_rows_loaded = len(bootstrap_data_from_file)
        print(f'number of rows loaded: {number_of_rows_loaded}')

        for index, row in enumerate(bootstrap_data_from_file):
            # excess values are ignored and will not be returned
            if index < len(bootstrap_data):
                bootstrap_data[index] = row

    # Generate more bootstrap data if required
    for index in range(number_of_bootstrap_samples):

        # If data has been loaded from file, do not over-write it
        if number_of_rows_loaded is not None:
            if index < number_of_rows_loaded:
                continue

        random_index = numpy.random.randint(len(eod_aapl_us_diff), size=len(eod_aapl_us_diff))
        bootstrap_sample = eod_aapl_us_diff[random_index]
        bootstrap_data[index] = bootstrap_sample

        #plot_bootstrap_data(iteration, bootstrap_sample)

        # number_of_bootstrap_samples = 100
        # step = 10
        step = (number_of_bootstrap_samples // 10)
        if step > 0:
            if index + 1 % step == 0:
                progress = float(index) / float(number_of_bootstrap_samples)
                print(f'progress: {round(progress * 100.0, 1)} %')

    numpy.savetxt(filename, bootstrap_data)

    print(f'bootstrap_data:\n{bootstrap_data}')
    return bootstrap_data


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


def parallel_function(x):

    #(index, data_x0) = x
    #(data, x0) = data_x0

    (index, data) = x

    #print(f'executing index {index}')

    #if index == 0:
    #    print(f'index={index}')
    #    print(f'len data={len(data)}')
    #    print(f'x0={x0}')

    amplitude = len(data)
    mean = data.mean()
    stddev = 1.1 * data.std()

    if amplitude != 250:
        print(f'amplitude={amplitude}')

    x0 = [amplitude, mean, stddev]

    optimize_result = perform_fitting_procedure(data, x0, index)

    #plot_optimize_result(data, index, optimize_result)

    if optimize_result is not None:
        fval = -optimize_result['fun']
        #ll_values[index] = fval
        return fval
    else:
        # save data for processing later
        numpy.savetxt(f'bootstrap_data_txt/data_{index}.txt', data)

    print(f'RETURN NONE!')
    return None


def plot_likelihood_distribution(ll_value_obs, ll_values):

    count = 0
    for ll_value in ll_values:
        if ll_value > ll_value_obs:
            count += 1
    cl_value = float(count) / float(len(ll_values))

    print(f'Number of values in histogram: {len(ll_values)}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Bootstrap log-likelihood value')
    ax.set_ylabel('Number of pseudo-experiments')

    (ll_bin_counts, ll_bin_edges, _) = ax.hist(ll_values, bins=100, histtype='step', color='tab:blue')

    x_index = numpy.digitize(ll_value_obs, ll_bin_edges)
    y_value = ll_bin_counts[x_index]
    #ax.plot([ll_value_obs, ll_value_obs], [0.0, y_value], 'k-')

    fill_x = ll_bin_edges[x_index:]
    fill_y = ll_bin_counts[x_index:]
    fill_y = numpy.insert(fill_y, 0, fill_y[0])
    ax.fill_between(fill_x, fill_y, step='pre', facecolor='none', edgecolor='tab:blue', hatch='///')

    s = f'CL = {round((1.0 - cl_value) * 100.0, ndigits=2)} %'
    ax.text(ll_value_obs - 1.0, y_value * 0.5, s, color='tab:blue', horizontalalignment='right')

    fig.savefig('ll.png')

    integral_1 = ll_bin_counts[:x_index].sum()
    integral_2 = ll_bin_counts[x_index:x_index+1].sum()
    integral_3 = ll_bin_counts[x_index+1:].sum()
    print(f'integrals: {integral_1}, {integral_2}, {integral_3}')
    print(f'lengths: {len(ll_bin_counts)}, {len(ll_bin_counts[:x_index])}, {len(ll_bin_counts[x_index:x_index+1])}, {len(ll_bin_counts[x_index+1:])}')
    print(f'cl integral: {float(integral_3 + integral_2) / float(integral_1 + integral_2)}')
    print(f'cl exact: {cl_value}')



def main():

    # Load AAPL
    eod_aapl_us = pandas.read_csv('./market_data/eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    # Load NVDA
    eod_nvda_us = pandas.read_csv('./market_data/eod_nvda_us.csv', dtype='str', delimiter=',')
    eod_nvda_us['Date'] = pandas.to_datetime(eod_nvda_us['Date'])
    eod_nvda_us['Close'] = eod_nvda_us['Close'].apply(lambda close: float(close))

    # Create difference
    eod_aapl_us['Diff'] = eod_aapl_us['Close'].diff()
    aapl_us_diff_std = eod_aapl_us['Diff'].std()

    plot_diff_with_stddev(eod_aapl_us['Diff'], aapl_us_diff_std)

    # Convert data from Pandas Dataframe to numpy array, drop NaN
    eod_aapl_us_diff = eod_aapl_us['Diff']
    eod_aapl_us_diff = eod_aapl_us_diff.dropna()
    eod_aapl_us_diff = eod_aapl_us_diff.to_numpy()

    # Input data is: eod_aapl_us_diff

    # used for plotting
    (bin_contents, bin_edges) = \
        numpy.histogram(eod_aapl_us_diff, bins=20)

    # fitting - maximum likelihood
    normalization_constant = bin_contents.sum()
    amplitude = normalization_constant #/ math.sqrt(2 * math.pi * aapl_us_diff_std * aapl_us_diff_std)
    mean = 0.0
    stddev = aapl_us_diff_std

    x0 = [amplitude, mean, stddev]

    optimize_result = perform_fitting_procedure(eod_aapl_us_diff, x0, None, True)

    # observed log-likelihood value
    ll_value_obs = -optimize_result['fun']

    # Create values of bin midpoints
    bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    normalization_constant = bin_contents.sum()
    normalization_constant = len(eod_aapl_us_diff)
    print(f'normalization_constant={normalization_constant}')
    #hist_model = normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)


    print('--- solution ---')
    print(optimize_result)
    print(f'amplitude={amplitude}')
    print(f'mean={mean}')
    print(f'stddev={stddev}')

    amplitude_out = optimize_result["x"][0]
    mean_out = optimize_result["x"][1]
    stddev_out = optimize_result["x"][2]
    print(f'amplitude_out={amplitude_out}')
    print(f'mean_out={mean_out}')
    print(f'stddev_out={stddev_out}')

    # compute the likelihood CL value
    bootstrap_data_count = 100000
    print(f'load bootstrap data')

    (ll_values, number_of_ll_values_loaded) = load_ll_values(bootstrap_data_count)

    print(f'Number of ll values loaded from disk: {number_of_ll_values_loaded}')
    print(f'Number of ll values required: {bootstrap_data_count}')
    print(f'')

    if number_of_ll_values_loaded < bootstrap_data_count:

        # need to calculate some more values
        bootstrap_data_to_process = bootstrap_data_count - number_of_ll_values_loaded

        print(f'File does not contain enough values')
        print(f'Number of additional ll values required: {len(bootstrap_data_to_process)}')

    # Plot the likelihood distribution figure
    plot_likelihood_distribution(ll_value_obs, ll_values)

    # end of likelihood CL value computation

    # plot the figure with optimized parameters
    # plot with errorbars
    fig = plt.figure()
    ax = fig.add_axes((.1, .3, .8, .6)) #fig.add_subplot(2, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change')
    ax.grid(True, axis='both', alpha=0.3, which='both')
    ax.set_xlim(xmin=-12, xmax=12)

    #curve_data_x_2 = numpy.linspace(-5 * stddev_out, 5 * stddev_out, 1000)
    curve_data_x = numpy.linspace(bin_edges[0], bin_edges[-1], 1000)
    normalization_constant = bin_contents.sum()
    curve_data_y = normalization_constant * stats.norm.pdf(curve_data_x, 0.0, aapl_us_diff_std)

    curve_data_x_2 = numpy.linspace(bin_edges[0], bin_edges[-1], 1000)
    curve_data_y_2 = amplitude_out * stats.norm.pdf(curve_data_x_2, mean_out, stddev_out)

    # stddev measurement model
    p1 = ax.plot(curve_data_x, curve_data_y, color='tab:green')

    # maximum likelihood model
    p2 = ax.plot(curve_data_x_2, curve_data_y_2, color='tab:orange')

    ax.legend(handles=[p1[0], p2[0]], labels=['Measured stddev', 'Maximum likelihood'])

    # data
    ax.errorbar(bin_midpoints, bin_contents, yerr=numpy.sqrt(bin_contents), \
        linestyle='none', marker='o', capsize=3, capthick=1, color='tab:blue')

    # model using stddev measurement
    hist_data_model_curve = normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)
    # model using maximum likelihood
    hist_data_model_curve_2 = amplitude_out * stats.norm.pdf(bin_midpoints, mean_out, stddev_out)

    ax2 = fig.add_axes((.1, .1, .8, .2))
    ax2.grid(True, axis='both', alpha=0.3, which='both')
    ax2.set_xlim(xmin=-12, xmax=12)

    ax2.plot([-12.0, 12.0], [0.0, 0.0], linewidth=2, color='black')

    xoffset = 0.1

    ax2.errorbar(bin_midpoints - xoffset, bin_contents - hist_data_model_curve, yerr=numpy.sqrt(bin_contents), \
        linestyle='none', marker='o', capsize=3, capthick=1, color='tab:green')

    ax2.errorbar(bin_midpoints + xoffset, bin_contents - hist_data_model_curve_2, yerr=numpy.sqrt(bin_contents), \
        linestyle='none', marker='o', capsize=3, capthick=1, color='tab:orange')

    fig.savefig('eod_aapl_us_diff_errorbar_likelihood_2.png')


if __name__ == '__main__':

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')
