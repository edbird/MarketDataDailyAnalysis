#!/usr/bin/env python3


import matplotlib
import matplotlib.pyplot as plt

import pandas
import math

import numpy
import scipy
import scipy.stats as stats

#from more_itertools import pairwise


def gaussian(x, amplitude, mean, stddev):

    y = numpy.power((x - mean) / stddev, 2.0)
    return amplitude * numpy.exp(-0.5 * y)


def optimize_function_least_squares(p, x, y):

    amplitude = p[0]
    mean = p[1]
    stddev = p[2]

    #print('optimize:')
    #print(amplitude)
    #print(mean)
    #print(stddev)
    #print(x)
    #print(y)

    model = gaussian(x, amplitude, mean, stddev)
    error = numpy.sqrt(y)
    residuals = numpy.zeros(len(y))
    for index in range(len(residuals)):
        if error[index] <= 0.0:
            pass
        else:
            value = (y[index] - model[index]) / error[index]
            residuals[index] = value
    #residuals = (y - model) / error

    return residuals


def optimize_function_likelihood(p, x, y):

    amplitude = p[0]
    mean = p[1]
    stddev = p[2]

    print('optimize:')
    print(amplitude)
    print(mean)
    print(stddev)
    print(x)
    print(y)

    # the change in price is expected to follow an exponential
    # distribution with a mean which is close to but not quite
    # zero
    model = gaussian(x, amplitude, mean, stddev)
    error = numpy.sqrt(y)

    fval = 0.0
    mval = 1.0

    # to create the likelihood function, for each bin midpoint (x)
    # get the value of the PDF, this gives the expected number of
    # events in this bin. Assume that the number of events in the
    # bin is distributed according to a Poisson distribution. The
    # total PDF for the whole histogram is a set of Poissons
    # multiplied together

    for index in range(len(model)):
        poisson_value = stats.poisson(model[index]).pmf(y[index])

        #print(f'poisson_value={poisson_value}')

        mval *= poisson_value

        # log likelihood
        fval += math.log(poisson_value)

    print(f'mval={mval}, fval={fval}')

    return -fval


def main():

    # Load AAPL
    eod_aapl_us = pandas.read_csv('eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    # Load NVDA
    eod_nvda_us = pandas.read_csv('eod_nvda_us.csv', dtype='str', delimiter=',')
    eod_nvda_us['Date'] = pandas.to_datetime(eod_nvda_us['Date'])
    eod_nvda_us['Close'] = eod_nvda_us['Close'].apply(lambda close: float(close))

    # calculate the parameters for a Gaussian Distribution
    aapl_us_var = eod_aapl_us['Close'].var()
    aapl_us_mean = eod_aapl_us['Close'].mean()
    aapl_us_mean = 0.0

    # Create difference timeseries
    eod_aapl_us['Diff'] = eod_aapl_us['Close'].diff()
    aapl_us_diff_std = eod_aapl_us['Diff'].std()

    # Histogram difference data
    range_low = eod_aapl_us['Diff'].min()
    range_high = eod_aapl_us['Diff'].max()
    range_low = math.floor(range_low)
    range_high = math.ceil(range_high)
    print(f'range: ({range_low}, {range_high})')
    (bin_counts, bin_edges) = numpy.histogram(eod_aapl_us['Diff'], bins=20, range=(range_low, range_high))

    curve_data_x = numpy.linspace(-5 * aapl_us_diff_std, 5 * aapl_us_diff_std, 1000)
    normalization_constant = bin_counts.sum()
    #normalization_constant_2 = 1.0 / (math.sqrt(2.0 * math.pi) * aapl_us_diff_std)
    curve_data_y = normalization_constant * stats.norm.pdf(curve_data_x, 0.0, aapl_us_diff_std)
    #curve_data_y = gaussian(curve_data_x, normalization_constant / math.sqrt(2 * math.pi * aapl_us_diff_std * aapl_us_diff_std), 0.0, aapl_us_diff_std)

    # Create values of bin midpoints
    bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    #hist_model = 250.0 / 236.0532057 * normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)
    hist_model = normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)

    print(f'normalization_constant={normalization_constant}')
    #print(f'normalization_constant_2={normalization_constant_2}')
    print(f'{len(bin_counts), len(hist_model)}')
    print(bin_counts)
    print(hist_model)


    ###########################################
    # fitting - chi square
    ###########################################

    amplitude = normalization_constant / math.sqrt(2 * math.pi * aapl_us_diff_std * aapl_us_diff_std)
    mean = 0.0
    stddev = aapl_us_diff_std
    #x0 = [-5.0 * aapl_us_diff_std, 5.0 * aapl_us_diff_std]
    x0 = [amplitude, mean, stddev]
    solution = scipy.optimize.least_squares(optimize_function_least_squares, x0, method='lm', \
        ftol=1.0e-8, xtol=1.0e-8, \
        max_nfev=1000000, args=(bin_midpoints, bin_counts)) # amplitude, mean, stddev

    print('--- solution ---')
    print(solution)
    print(f'amplitude={amplitude}')
    print(f'mean={mean}')
    print(f'stddev={stddev}')

    amplitude_out_lsq = solution["x"][0]
    mean_out_lsq = solution["x"][1]
    stddev_out_lsq = solution["x"][2]
    print(f'amplitude_out_lsq={amplitude_out_lsq}')
    print(f'mean_out_lsq={mean_out_lsq}')
    print(f'stddev_out_lsq={stddev_out_lsq}')

    #residuals = optimize_function(p=solution['x'], x=bin_midpoints, y=bin_counts)
    residuals = solution.fun
    s_sq = numpy.power(residuals, 2.0).sum()
    print(f's_sq={s_sq}')
    print(f's_sq/ndf={s_sq/(len(bin_counts) - 3)}')

    cost = solution.cost
    optimality = solution.optimality
    print(f'2*cost={2.0 * cost}, optimality={optimality}')

    #s_square = (solution['fvec'] ** 2.0).sum() / (len(solution['fvec']) - len(solution[0]))

    # chi2
    #(chi2, chi2_p) = scipy.stats.chisquare(bin_counts, hist_model, 2)

    #print(f'chi2={chi2}, chi2_p={chi2_p}')

    ndf = len(bin_counts) - len(x0)
    print(ndf)
    print(1.0 - stats.chi2.cdf(2.0 * cost, ndf))

    ###########################################
    # end
    ###########################################


    # fitting - maximum likelihood
    amplitude = normalization_constant / math.sqrt(2 * math.pi * aapl_us_diff_std * aapl_us_diff_std)
    mean = 0.0
    stddev = aapl_us_diff_std
    #x0 = [-5.0 * aapl_us_diff_std, 5.0 * aapl_us_diff_std]
    x0 = [amplitude, mean, stddev]
    optimize_result = scipy.optimize.minimize(optimize_function_likelihood, x0, method='BFGS', \
        args=(bin_midpoints, bin_counts), \
        options={'disp': True})

    print('--- solution ---')
    print(optimize_result)
    print(f'amplitude={amplitude}')
    print(f'mean={mean}')
    print(f'stddev={stddev}')

    amplitude_out_ml = optimize_result["x"][0]
    mean_out_ml = optimize_result["x"][1]
    stddev_out_ml = optimize_result["x"][2]
    print(f'amplitude_out_ml={amplitude_out_ml}')
    print(f'mean_out_ml={mean_out_ml}')
    print(f'stddev_out_ml={stddev_out_ml}')

    #residuals = optimize_function(p=solution['x'], x=bin_midpoints, y=hist_data)
    residuals = optimize_result.fun
    s_sq = numpy.power(residuals, 2.0).sum()
    print(f's_sq={s_sq}')
    print(f's_sq/ndf={s_sq/(len(bin_counts) - 3)}')

    #cost = optimize_result.cost
    #optimality = optimize_result.optimality
    #print(f'2*cost={2.0 * cost}, optimality={optimality}')

    #s_square = (solution['fvec'] ** 2.0).sum() / (len(solution['fvec']) - len(solution[0]))

    # chi2
    #(chi2, chi2_p) = scipy.stats.chisquare(hist_data, hist_model, 2)

    #print(f'chi2={chi2}, chi2_p={chi2_p}')

    #ndf = len(hist_data) - len(x0)
    #print(ndf)
    #print(1.0 - stats.chi2.cdf(2.0 * cost, ndf))

    # plot the figure with optimized parameters
    # plot with errorbars
    fig = plt.figure()
    ax = fig.add_axes((.1, .3, .8, .6)) #fig.add_subplot(2, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change')
    #fig.autofmt_xdate(rotation=90)
    ax.grid(True, axis='both', alpha=0.3, which='both')

    curve_data_x_lsq = numpy.linspace(-5 * stddev_out_lsq, 5 * stddev_out_lsq, 1000)
    curve_data_x_ml = numpy.linspace(-5 * stddev_out_lsq, 5 * stddev_out_lsq, 1000)

    curve_data_y_lsq = gaussian(curve_data_x, amplitude_out_lsq, mean_out_lsq, stddev_out_lsq)
    curve_data_y_ml = gaussian(curve_data_x, amplitude_out_ml, mean_out_ml, stddev_out_ml)

    p1a = ax.plot(curve_data_x_lsq, curve_data_y_lsq, color='tab:orange')
    p1b = ax.plot(curve_data_x_ml, curve_data_y_ml, color='tab:red')
    # pre-optimized model
    p1c = ax.plot(curve_data_x, curve_data_y, color='tab:green')
    p1d = ax.errorbar(bin_midpoints, bin_counts, yerr=numpy.sqrt(bin_counts), \
        linestyle='none', marker='_', capsize=3, capthick=1, color='tab:blue')
    ax.set_xlim(xmin=-12, xmax=12)

    hist_data_model_curve_lsq = gaussian(bin_midpoints, amplitude_out_lsq, mean_out_lsq, stddev_out_lsq)
    hist_data_model_curve_ml = gaussian(bin_midpoints, amplitude_out_ml, mean_out_ml, stddev_out_ml)
    #hist_data_model_curve = normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)

    ax.legend(handles=[p1a[0], p1b[0], p1c[0], p1d[0]], \
        labels=['Least Squares', 'Maximum Likelihood', 'Measured Parameters', 'Data'],
        loc='upper left')

    # Second Axis
    ax2 = fig.add_axes((.1, .1, .8, .2))
    p2a = ax2.plot([-12.0, 12.0], [0.0, 0.0], linewidth=2, color='black')
    p2b = ax2.errorbar(bin_midpoints, bin_counts - hist_data_model_curve_lsq, yerr=numpy.sqrt(bin_counts), \
        linestyle='none', marker='_', capsize=3, capthick=1, color='tab:orange')
    p2c = ax2.errorbar(bin_midpoints, bin_counts - hist_data_model_curve_ml, yerr=numpy.sqrt(bin_counts), \
        linestyle='none', marker='_', capsize=3, capthick=1, color='tab:red')
    ax2.grid(True, axis='both', alpha=0.3, which='both')
    ax2.set_xlim(xmin=-12, xmax=12)

    ax2.legend(handles=[p2b[0], p2c[0]], \
        labels=['Data - Least Squares', 'Data - Maximum Likelihood'],
        loc='upper left', bbox_to_anchor=(0.7,1), fontsize=8)


    fig.savefig('eod_aapl_us_diff_errorbar_sql_ml.png')



if __name__ == '__main__':
    main()
