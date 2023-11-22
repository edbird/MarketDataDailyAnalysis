#!/usr/bin/env python3

from optimize_least_squares import optimize_function_least_squares
from optimize_maximum_likelihood import optimize_function_maximum_likelihood_unbinned
from optimize_maximum_likelihood import optimize_function_maximum_likelihood_binned

from libloaddata import load_diff_data_aapl

import math
import numpy

import scipy
import scipy.stats as stats

import matplotlib.pyplot as plt




def main():

    # load data
    eod_aapl_us_diff = load_diff_data_aapl()

    #eod_aapl_us_diff = eod_aapl_us_diff[(eod_aapl_us_diff >= -3) & (eod_aapl_us_diff <= 3)]

    # calculate the parameters for a Gaussian Distribution
    aapl_us_diff_mean = eod_aapl_us_diff.mean()
    aapl_us_diff_std = eod_aapl_us_diff.std()
    parameters_measured = (aapl_us_diff_mean, aapl_us_diff_std)

    # Histogram difference data
    range_low = eod_aapl_us_diff.min()
    range_high = eod_aapl_us_diff.max()
    range_low = math.floor(range_low)
    range_high = math.ceil(range_high)
    print(f'range: ({range_low}, {range_high})')
    (bin_counts, bin_edges) = numpy.histogram(eod_aapl_us_diff, bins=20, range=(range_low, range_high))

    # Create values of bin midpoints
    bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])


    parameters_least_squares = \
        do_least_squares_optimization(bin_counts, bin_midpoints, aapl_us_diff_mean, aapl_us_diff_std)

    parameters_maximum_likelihood_unbinned = \
        do_maximum_likelihood_optimization_unbinned(eod_aapl_us_diff, aapl_us_diff_mean, aapl_us_diff_std)

    #(bin_counts_2, bin_edges_2) = numpy.histogram(eod_aapl_us_diff, bins=20, range=(range_low, range_high))
    #bin_midpoints_2 = 0.5 * (bin_edges_2[1:] + bin_edges_2[:-1])\

    bin_counts_2 = bin_counts
    bin_midpoints_2 = bin_midpoints

    parameters_maximum_likelihood_binned = \
        do_maximum_likelihood_optimization_binned(bin_counts_2, bin_midpoints_2, aapl_us_diff_mean, aapl_us_diff_std)

    # fitting - maximum likelihood - TODO

    plot_figure(bin_counts, bin_midpoints,
                parameters_measured,
                parameters_least_squares,
                parameters_maximum_likelihood_unbinned,
                parameters_maximum_likelihood_binned)


def plot_figure(bin_counts, bin_midpoints,
                parameters_measured,
                parameters_least_squares,
                parameters_maximum_likelihood,
                parameters_maximum_likelihood_binned):

    (aapl_us_diff_mean, aapl_us_diff_std) = parameters_measured
    (mean_out_lsq, stddev_out_lsq) = parameters_least_squares
    (mean_out_ml, stddev_out_ml) = parameters_maximum_likelihood
    (mean_out_ml_binned, stddev_out_ml_binned) = parameters_maximum_likelihood_binned

    # create smooth model using measured mean and variance
    normalization_constant = bin_counts.sum()

    curve_data_x =      numpy.linspace(-5 * aapl_us_diff_std, 5 * aapl_us_diff_std, 1000)
    #curve_data_x_lsq =  numpy.linspace(-5 * stddev_out_lsq, 5 * stddev_out_lsq, 1000)
    #curve_data_x_ml =   numpy.linspace(-5 * stddev_out_ml, 5 * stddev_out_ml, 1000)

    curve_data_y =      normalization_constant * stats.norm.pdf(curve_data_x, loc=aapl_us_diff_mean, scale=aapl_us_diff_std)
    curve_data_y_lsq =  normalization_constant * stats.norm.pdf(curve_data_x, loc=mean_out_lsq, scale=stddev_out_lsq)
    curve_data_y_ml =   normalization_constant * stats.norm.pdf(curve_data_x, loc=mean_out_ml, scale=stddev_out_ml)
    curve_data_y_ml_binned =   normalization_constant * stats.norm.pdf(curve_data_x, loc=mean_out_ml_binned, scale=stddev_out_ml_binned)

    hist_data_model_curve_lsq = normalization_constant * stats.norm.pdf(bin_midpoints, loc=mean_out_lsq, scale=stddev_out_lsq)
    hist_data_model_curve_ml =  normalization_constant * stats.norm.pdf(bin_midpoints, loc=mean_out_ml, scale=stddev_out_ml)
    hist_data_model_curve_ml_binned =  normalization_constant * stats.norm.pdf(bin_midpoints, loc=mean_out_ml_binned, scale=stddev_out_ml_binned)


    # plot the figure with optimized parameters
    # plot with errorbars
    fig = plt.figure()


    # First Axis
    ax = fig.add_axes((.1, .3, .8, .6)) #fig.add_subplot(2, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change')
    ax.grid(True, axis='both', alpha=0.3, which='both')
    ax.set_xlim(xmin=-12, xmax=12)

    #p1c = ax.plot(curve_data_x, curve_data_y, color='tab:green')
    p1a = ax.plot(curve_data_x, curve_data_y_lsq, color='tab:orange')
    p1e = ax.plot(curve_data_x, curve_data_y_ml_binned, color='tab:green')
    p1b = ax.plot(curve_data_x, curve_data_y_ml, color='tab:red')

    p1d = ax.errorbar(bin_midpoints, bin_counts, yerr=numpy.sqrt(bin_counts), \
        linestyle='none', marker='_', capsize=3, capthick=1, color='tab:blue')

    #ax.legend(handles=[p1a[0], p1b[0], p1c[0], p1d[0]], \
    #    labels=['Least Squares', 'Maximum Likelihood', 'Measured Parameters', 'Data'],
    #    loc='upper left')
    ax.legend(handles=[p1a[0], p1b[0], p1e[0], p1d[0]], \
        labels=['Least Squares (LS)', 'Maximum Likelihood (ML)', 'Histogram ML', 'Data'],
        loc='upper left', fontsize=9)


    # Second Axis
    ax2 = fig.add_axes((.1, .1, .8, .2))
    ax2.grid(True, axis='both', alpha=0.3, which='both')
    ax2.set_xlim(xmin=-12, xmax=12)
    ax2.set_ylim(ymin=-20, ymax=30)

    p2a = ax2.plot([-12.0, 12.0], [0.0, 0.0], linewidth=1, color='black')

    p2b = ax2.errorbar(bin_midpoints, bin_counts - hist_data_model_curve_lsq, yerr=numpy.sqrt(bin_counts), \
        linestyle='none', marker='_', capsize=3, capthick=1, color='tab:orange')

    p2c = ax2.errorbar(bin_midpoints, bin_counts - hist_data_model_curve_ml, yerr=numpy.sqrt(bin_counts), \
        linestyle='none', marker='_', capsize=3, capthick=1, color='tab:red')

    ax2.legend(handles=[p2b[0], p2c[0]], \
        labels=['Data - LS', 'Data - ML'],
        loc='upper left', bbox_to_anchor=(0.0,1), fontsize=8)


    # Output
    fig.savefig('eod_aapl_us_diff_errorbar_lsq_ml.png')
    fig.savefig('eod_aapl_us_diff_errorbar_lsq_ml.pdf')




def do_least_squares_optimization(bin_counts, bin_midpoints, aapl_us_diff_mean, aapl_us_diff_std):

    ###########################################
    # fitting - chi square
    ###########################################

    normalization_constant = bin_counts.sum()
    #hist_model = normalization_constant * stats.norm.pdf(bin_midpoints, 0.0, aapl_us_diff_std)

    #amplitude = normalization_constant / math.sqrt(2 * math.pi * aapl_us_diff_std * aapl_us_diff_std)
    amplitude = normalization_constant
    mean = aapl_us_diff_mean
    stddev = aapl_us_diff_std
    x0 = [mean, stddev]
    solution = \
        scipy.optimize.least_squares(
            optimize_function_least_squares,
            x0,
            method='lm',
            ftol=1.0e-8,
            xtol=1.0e-8,
            max_nfev=1000000,
            args=(bin_midpoints, bin_counts, amplitude))

    print('')
    print('**** Least Squares | Chi-Square ****')
    print('--- solution ---')
    print(solution)
    print(f'amplitude={amplitude}')
    print(f'mean={mean}')
    print(f'stddev={stddev}')

    mean_out = solution["x"][0]
    stddev_out = solution["x"][1]
    print(f'mean_out_lsq={mean_out}')
    print(f'stddev_out_lsq={stddev_out}')

    #residuals = optimize_function(p=solution['x'], x=bin_midpoints, y=bin_counts)
    residuals = solution.fun
    s_sq = numpy.power(residuals, 2.0).sum()
    print(f's_sq={s_sq}')
    ndf = len(bin_counts) - len(x0)
    print(f's_sq/ndf={s_sq / float(ndf)}')

    cost = solution.cost
    optimality = solution.optimality
    print(f'2*cost={2.0 * cost}, optimality={optimality}')

    #s_square = (solution['fvec'] ** 2.0).sum() / (len(solution['fvec']) - len(solution[0]))

    # chi2
    #(chi2, chi2_p) = scipy.stats.chisquare(bin_counts, hist_model, 2)

    #print(f'chi2={chi2}, chi2_p={chi2_p}')

    print(ndf)
    print(1.0 - stats.chi2.cdf(2.0 * cost, ndf))

    ###########################################
    # end
    ###########################################

    return (mean_out, stddev_out)


def do_maximum_likelihood_optimization_unbinned(data, aapl_us_diff_mean, aapl_us_diff_std):

    ###########################################
    # fitting - maximum likelihood unbinned
    ###########################################

    mean = aapl_us_diff_mean
    stddev = aapl_us_diff_std
    #x0 = [-5.0 * aapl_us_diff_std, 5.0 * aapl_us_diff_std]
    x0 = [mean, stddev]
    optimize_result = \
        scipy.optimize.minimize(
            optimize_function_maximum_likelihood_unbinned,
            x0,
            method='BFGS',
            args=(data),
            options={'disp': True})

    print('')
    print('**** Maximum Likelihood | Unbinned ****')
    print('--- solution ---')
    print(optimize_result)
    print(f'mean={mean}')
    print(f'stddev={stddev}')

    mean_out = optimize_result["x"][0]
    stddev_out = optimize_result["x"][1]
    print(f'mean_out_ml={mean_out}')
    print(f'stddev_out_ml={stddev_out}')

    #residuals = optimize_function(p=solution['x'], x=bin_midpoints, y=hist_data)
    residuals = optimize_result.fun
    s_sq = numpy.power(residuals, 2.0).sum()
    print(f's_sq={s_sq}')
    ndf = len(data) - len(x0)
    print(f's_sq/ndf={s_sq / float(ndf)}')

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

    return (mean_out, stddev_out)


def do_maximum_likelihood_optimization_binned(bin_counts, bin_midpoints, aapl_us_diff_mean, aapl_us_diff_std):

    ###########################################
    # fitting - maximum likelihood binned
    ###########################################

    amplitude = bin_counts.sum()

    mean = aapl_us_diff_mean
    stddev = aapl_us_diff_std
    x0 = [mean, stddev]

    optimize_result = \
        scipy.optimize.minimize(
            optimize_function_maximum_likelihood_binned,
            x0,
            method='BFGS',
            args=(bin_midpoints, bin_counts, amplitude),
            options={'disp': True})

    print('')
    print('**** Maximum Likelihood | Binned ****')
    print('--- solution ---')
    print(optimize_result)
    print(f'mean={mean}')
    print(f'stddev={stddev}')

    mean_out = optimize_result["x"][0]
    stddev_out = optimize_result["x"][1]
    print(f'mean_out_ml={mean_out}')
    print(f'stddev_out_ml={stddev_out}')

    #residuals = optimize_function(p=solution['x'], x=bin_midpoints, y=hist_data)
    residuals = optimize_result.fun
    s_sq = numpy.power(residuals, 2.0).sum()
    print(f's_sq={s_sq}')
    ndf = len(bin_counts) - len(x0)
    print(f's_sq/ndf={s_sq / float(ndf)}')

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

    return (mean_out, stddev_out)





if __name__ == '__main__':
    main()
