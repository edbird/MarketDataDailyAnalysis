#!/usr/bin/env python3


import matplotlib
import matplotlib.pyplot as plt

import pandas
import math

import numpy
import scipy
import scipy.stats as stats

from more_itertools import pairwise


def gaussian(x, amplitude, mean, stddev):

    y = numpy.power((x - mean) / stddev, 2.0)
    return amplitude * numpy.exp(-0.5 * y)


def optimize_function(p, x, y):

    amplitude = p[0]
    mean = p[1]
    stddev = p[2]

    print('optimize:')
    print(amplitude)
    print(mean)
    print(stddev)
    print(x)
    print(y)

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


def main():

    # Load AAPL
    eod_aapl_us = pandas.read_csv('eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    # Load NVDA
    eod_nvda_us = pandas.read_csv('eod_nvda_us.csv', dtype='str', delimiter=',')
    eod_nvda_us['Date'] = pandas.to_datetime(eod_nvda_us['Date'])
    eod_nvda_us['Close'] = eod_nvda_us['Close'].apply(lambda close: float(close))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    #fig.autofmt_xdate(rotation=90)
    ax.grid(True, axis='both', alpha=0.3, which='both')

    ax.scatter(eod_aapl_us['Date'], eod_aapl_us['Close'], label='AAPL')
    ax.scatter(eod_nvda_us['Date'], eod_nvda_us['Close'], label='NVDA')
    plt.legend(loc='upper left')
    fig.savefig('eod_aapl_nvda_us.png')

    aapl_us_var = eod_aapl_us['Close'].var()
    nvda_us_var = eod_nvda_us['Close'].var()
    aapl_us_std = eod_aapl_us['Close'].std()
    nvda_us_std = eod_nvda_us['Close'].std()
    aapl_us_mean = eod_aapl_us['Close'].mean()
    nvda_us_mean = eod_nvda_us['Close'].mean()

    print(f'Var: AAPL={aapl_us_var}, NVDA={nvda_us_var}')
    print(f'Std: AAPL={aapl_us_std}, NVDA={nvda_us_std}')
    print(f'Mean: AAPL={aapl_us_mean}, NVDA={nvda_us_mean}')

    sum = 0.0
    sum2 = 0.0
    count = 0
    last = None
    for close in eod_aapl_us['Close']:

        #print(f'sum2={sum2}')

        if last is None:
            count += 1
            sum += close
            sum2 += close * close
            last = close
        else:
            count += 1
            diff = close - last
            sum += close
            sum2 += close * close
            last = close

    #print(f'sum2={sum2}, sum={sum}, sum*sum={sum*sum}')
    mean = sum / float(count)
    var = (sum2) / float(count - 1)  - (float(count) / float(count - 1)) * mean * mean
    var = (sum2) / float(count - 1)  - (1.0 / float(count) / float(count - 1)) * sum * sum

    print(f'N={count}, var={var}, sum2={sum2}, mean={mean}, mean^2={mean*mean}')


    # alternative calculation
    # slightly different result but compatiable
    var_2 = 0.0
    for close in eod_aapl_us['Close']:

        x = abs(close - mean) ** 2.0
        var_2 += x

    var_2 = var_2 / float(count - 1)
    print(f'var_2={var_2}')


    # Create difference
    eod_aapl_us['Diff'] = eod_aapl_us['Close'].diff()
    aapl_us_diff_std = eod_aapl_us['Diff'].std()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change')
    #fig.autofmt_xdate(rotation=90)
    ax.grid(True, axis='both', alpha=0.3, which='both')

    (n, bins, patches) = ax.hist(eod_aapl_us['Diff'], label='AAPL', bins=20, density=False)
    curve_data_x = numpy.linspace(-5 * aapl_us_diff_std, 5 * aapl_us_diff_std, 1000)
    normalization_constant = n.sum()
    #normalization_constant_2 = 1.0 / (math.sqrt(2.0 * math.pi) * aapl_us_diff_std)
    curve_data_y = normalization_constant * stats.norm.pdf(curve_data_x, 0.0, aapl_us_diff_std)
    #curve_data_y = gaussian(curve_data_x, normalization_constant / math.sqrt(2 * math.pi * aapl_us_diff_std * aapl_us_diff_std), 0.0, aapl_us_diff_std)
    ax.plot(curve_data_x, curve_data_y)
    fig.savefig('eod_aapl_us_diff.png')

    eod_aapl_us['ll_p'] = stats.norm.pdf(eod_aapl_us['Diff'], 0.0, aapl_us_diff_std)
    ll = eod_aapl_us['ll_p'].prod()
    print(f'll={ll}')

    bin_midpoints = 0.5 * (bins[1:] + bins[:-1])
    #print(f'n={n}, len(n)={len(n)}')
    #print(f'bins={bins}, len(bins)={len(bins)}')
    #print(f'bin_midpoints={bin_midpoints}, len(bin_midpoints)={len(bin_midpoints)}')

    hist_data = n
    #hist_model = 250.0 / 236.0532057 * normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)
    hist_model = normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)
    ax.plot(bin_midpoints, hist_model)
    fig.savefig('eod_aapl_us_diff_2.png')

    print(f'normalization_constant={normalization_constant}')
    #print(f'normalization_constant_2={normalization_constant_2}')
    print(f'{len(hist_data), len(hist_model)}')
    print(hist_data)
    print(hist_model)

    print(f'sum of frequencies: {hist_data.sum()}, {hist_model.sum()}')

    # fitting
    amplitude = normalization_constant / math.sqrt(2 * math.pi * aapl_us_diff_std * aapl_us_diff_std)
    mean = 0.0
    stddev = aapl_us_diff_std
    #x0 = [-5.0 * aapl_us_diff_std, 5.0 * aapl_us_diff_std]
    x0 = [amplitude, mean, stddev]
    solution = scipy.optimize.least_squares(optimize_function, x0, method='lm', \
        ftol=1.0e-8, xtol=1.0e-8, \
        max_nfev=1000000, args=(bin_midpoints, hist_data)) # amplitude, mean, stddev

    print('--- solution ---')
    print(solution)
    print(f'amplitude={amplitude}')
    print(f'mean={mean}')
    print(f'stddev={stddev}')

    amplitude_out = solution["x"][0]
    mean_out = solution["x"][1]
    stddev_out = solution["x"][2]
    print(f'amplitude_out={amplitude_out}')
    print(f'mean_out={mean_out}')
    print(f'stddev_out={stddev_out}')

    #residuals = optimize_function(p=solution['x'], x=bin_midpoints, y=hist_data)
    residuals = solution.fun
    s_sq = numpy.power(residuals, 2.0).sum()
    print(f's_sq={s_sq}')
    print(f's_sq/ndf={s_sq/(len(hist_data) - 3)}')

    cost = solution.cost
    optimality = solution.optimality
    print(f'2*cost={2.0 * cost}, optimality={optimality}')

    #s_square = (solution['fvec'] ** 2.0).sum() / (len(solution['fvec']) - len(solution[0]))

    # chi2
    #(chi2, chi2_p) = scipy.stats.chisquare(hist_data, hist_model, 2)

    #print(f'chi2={chi2}, chi2_p={chi2_p}')

    ndf = len(hist_data) - len(x0)
    print(ndf)
    print(1.0 - stats.chi2.cdf(2.0 * cost, ndf))

    # plot the figure with optimized parameters
    # plot with errorbars
    fig = plt.figure()
    ax = fig.add_axes((.1, .3, .8, .6)) #fig.add_subplot(2, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change')
    #fig.autofmt_xdate(rotation=90)
    ax.grid(True, axis='both', alpha=0.3, which='both')

    #(n, bins, patches) = ax.hist(eod_aapl_us['Diff'], label='AAPL', bins=20, density=False)
    curve_data_x_2 = numpy.linspace(-5 * stddev_out, 5 * stddev_out, 1000)
    #curve_data_y_2 = amplitude_out * stats.norm.pdf(curve_data_x, mean_out, stddev_out)
    curve_data_y_2 = gaussian(curve_data_x, amplitude_out, mean_out, stddev_out)

    ax.plot(curve_data_x_2, curve_data_y_2, color='tab:orange')
    # pre-optimized model
    ax.plot(curve_data_x, curve_data_y, color='tab:green')
    ax.errorbar(bin_midpoints, hist_data, yerr=numpy.sqrt(hist_data), \
        linestyle='none', marker='o', capsize=3, capthick=1, color='tab:blue')
    ax.set_xlim(xmin=-12, xmax=12)

    hist_data_model_curve_2 = gaussian(bin_midpoints, amplitude_out, mean_out, stddev_out)
    hist_data_model_curve = normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)

    ax2 = fig.add_axes((.1, .1, .8, .2))
    ax2.plot([-12.0, 12.0], [0.0, 0.0], linewidth=2, color='black')
    ax2.errorbar(bin_midpoints, hist_data - hist_data_model_curve_2, yerr=numpy.sqrt(hist_data), \
        linestyle='none', marker='o', capsize=3, capthick=1, color='tab:blue')
    ax2.grid(True, axis='both', alpha=0.3, which='both')
    ax2.set_xlim(xmin=-12, xmax=12)

    fig.savefig('eod_aapl_us_diff_errorbar.png')



if __name__ == '__main__':
    main()
