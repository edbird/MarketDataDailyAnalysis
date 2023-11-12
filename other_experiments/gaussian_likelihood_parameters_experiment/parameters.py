#!/usr/bin/env python3


import math
import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy



def calculate_likelihood(data, mean, stddev):
    
    pdf_values = scipy.stats.norm.pdf(data, mean, stddev)
    #print(f'mean={mean}, stddev={stddev}')
    #print(f'pdf: {pdf_values}')
    likelihood = pdf_values.prod()
    return likelihood


def calculate_log_likelihood(data, mean, stddev):
    
    pdf_values = scipy.stats.norm.pdf(data, mean, stddev)
    log_pdf_values = numpy.log(pdf_values)
    #print(f'data: {data}')
    #print(f'pdf_values: {pdf_values}')
    #print(f'log_pdf_values: {log_pdf_values}')
    log_likelihood = log_pdf_values.sum()
    return log_likelihood


def optimize_function(p, data):
    
    mean = p[0]
    stddev = p[1]  

    #return -calculate_likelihood(data, mean, stddev)
    return -calculate_log_likelihood(data, mean, stddev)


def main():
    
    # draw a batch of numbers from a Gaussian distribution
    mean = 0.0
    stddev = 1.0
    batch_size = 10
    numpy.random.seed(5)
    numpy.random.seed(500)
    random_numbers = numpy.random.normal(mean, stddev, batch_size)
    
    # plot random numbers, using fine histogram binning
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.hist(random_numbers, bins=200, range=(-5 * stddev, 5 * stddev))
    figure.savefig('gaussian_batch.png', dpi=300)
    
    measured_mean = random_numbers.mean()
    measured_stddev = random_numbers.std()
    
    print(f'{measured_mean}')
    print(f'{measured_stddev}')

    #x0 = [measured_mean, measured_stddev]    
    x0 = [0.0, 1.0]    
    optimize_result = scipy.optimize.minimize(optimize_function, x0, method='BFGS', args=(random_numbers), options={'disp': True}, tol=1.0e-10)
    likelihood_value = -optimize_result['fun']
    optimized_mean = optimize_result['x'][0]
    optimized_stddev = optimize_result['x'][1]
    
    print(f'{0.0}, {1.0}, {calculate_log_likelihood(random_numbers, 0.0, 1.0)}')
    print(f'{measured_mean}, {measured_stddev}, {calculate_log_likelihood(random_numbers, measured_mean, measured_stddev)}')
    print(f'{optimized_mean}, {optimized_stddev}, {calculate_log_likelihood(random_numbers, optimized_mean, optimized_stddev)}')
    
    # plot results
    figure=plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    x = numpy.linspace(-5 * stddev, 5 * stddev, 1000)
    p1 = ax.plot(x, batch_size * scipy.stats.norm.pdf(x, 0.0, 1.0))
    p2 = ax.plot(x, batch_size * scipy.stats.norm.pdf(x, measured_mean, measured_stddev))
    p3 = ax.plot(x, batch_size * scipy.stats.norm.pdf(x, optimized_mean, optimized_stddev))
    (_, _, p4) = ax.hist(random_numbers, bins=200, range=(-5 * stddev, 5 * stddev))
    ax.legend(handles=[p1[0], p2[0], p3[0], p4[0]], labels=['original', 'measured', 'optimized', 'data'])
    figure.savefig('gaussian_batch_fit_results.png', dpi=300)
    
    
if __name__ == '__main__':
    main()