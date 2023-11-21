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


from plot_aapl_likelihood_plotting_module import plot_failed_fit
from plot_aapl_likelihood_plotting_module import plot_bootstrap_data

from plot_aapl_likelihood.plot_aapl_likelihood.optimize_module_binned import OptimizeFunctionException, perform_fitting_procedure


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
    bootstrap_data_count = 1900000
    print(f'generating bootstrap data')
    bootstrap_data = generate_bootstrap_data(eod_aapl_us_diff, bootstrap_data_count)

    (ll_values, number_of_ll_values_loaded) = load_ll_values(len(bootstrap_data))

    print(f'Number of ll values loaded from disk: {number_of_ll_values_loaded}')
    print(f'Number of ll values required: {len(bootstrap_data)}')
    print(f'')

    if number_of_ll_values_loaded < len(bootstrap_data):

        # need to calculate some more values

        # chop of some of the bootstrap data if some ll values were loaded, if not process
        # the whole data
        bootstrap_data_to_process = \
            bootstrap_data[number_of_ll_values_loaded:] #if number_of_ll_values_loaded is not None else bootstrap_data

        print(f'Number of ll values to generate: {len(bootstrap_data_to_process)}')

        with Pool(processes=10) as pool:

            # NOTE: now estimate x0 each time to improve chance of convergence
            #bootstrap_data_x0 = []
            #for each_bootstrap_data in bootstrap_data:
            #    bootstrap_data_x0.append((each_bootstrap_data, x0))

            print(f'executing pool')

            #return_values = pool.map(parallel_function, enumerate(bootstrap_data_x0) )
            index_offset = number_of_ll_values_loaded #if number_of_ll_values_loaded is not None else 0
            return_values = pool.map(parallel_function, enumerate(bootstrap_data_to_process, index_offset))

            for index, ll_value in enumerate(return_values):

                if ll_value is None:
                    ll_value = numpy.nan

                ll_values[index + index_offset] = ll_value

    #for index, data in enumerate(bootstrap_data):
    #    optimize_result = perform_fitting_procedure(data, x0)
    #    fval = -optimize_result['fun']
    #    ll_values[index] = fval
    #
    #    progress = float(index) / float(bootstrap_data_count)
    #    if index % bootstrap_data_count / 10 == 0:
    #        print(f'progress={progress}')

    #print(f'{ll_values}')
    ll_values = ll_values[~numpy.isnan(ll_values)]
    #print(f'{ll_values}')

    # save ll values
    save_ll_values(ll_values)

    # end of likelihood CL value computation


if __name__ == '__main__':

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')
