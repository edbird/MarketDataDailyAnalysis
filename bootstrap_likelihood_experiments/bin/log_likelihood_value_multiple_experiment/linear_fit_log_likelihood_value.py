#!/usr/bin/env python3


import numpy
import datetime
from datetime import timezone

import matplotlib.pyplot as plt

import scipy



def optimize_function_least_squares(params, x, y, e):

    m = params[0]
    c = params[1]

    model = m * x + c

    residuals = (y - model) / e

    return residuals


def main():
    
    data = numpy.load('values_to_save.npy')
    
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Number of samples per experiment')
    axis.set_ylabel('Optimized log-likelihood value')
    
    x = data[:, 0]
    y = data[:, 1]
    e = data[:, 2]
    
    x0 = [0.0, 0.0]
    
    optimize_result = scipy.optimize.least_squares(
            optimize_function_least_squares,
            x0,
            method='lm',
            ftol=1.0e-8,
            xtol=1.0e-8,
            max_nfev=1000000,
            args=(x, y, e))
    
    print(optimize_result)
    
    m_out = optimize_result.x[0]
    c_out = optimize_result.x[1]
    
    x_plot = [x[0], x[-1]]
    x_plot = numpy.array(x_plot)
    y_plot = m_out * x_plot + c_out
    
    axis.errorbar(x, y, e, ls='none', fmt='o', capsize=3)
    axis.plot(x_plot, y_plot)
    
    figure.savefig('linear_fit.pdf')
    figure.savefig('linear_fit.png')
    
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Number of samples per experiment')
    axis.set_ylabel('Optimized log-likelihood value residuals')
    
    y_model = m_out * x + c_out
    
    axis.errorbar(x, y - y_model, e, ls='none', fmt='o', capsize=3)
    axis.plot(x_plot, y_plot - y_plot)
    
    figure.savefig('linear_fit_residuals.pdf')
    figure.savefig('linear_fit_residuals.png')
    

def timed_main():

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':
    timed_main()
    