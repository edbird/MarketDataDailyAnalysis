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
    
    fig.savefig('plot_aapl_likelihood/eod_aapl_us_diff.png')


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

    
# NOTE: this won't actually work because I don't supply an index
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
    
    fig.savefig(f'plot_aapl_likelihood/failed_fit/failed_fit_{index}.png')
    plt.close() 
    
    
def main():

    # Load AAPL
    eod_aapl_us = pandas.read_csv('eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    # Load NVDA
    eod_nvda_us = pandas.read_csv('eod_nvda_us.csv', dtype='str', delimiter=',')
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

    fig.savefig('plot_aapl_likelihood/eod_aapl_us_diff_errorbar_likelihood.png')


if __name__ == '__main__':
    
    time_start = datetime.datetime.now(timezone.utc)
    
    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')
