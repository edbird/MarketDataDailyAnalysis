#!/usr/bin/env python3

from libexperiment import generate_experiment_data_gaussian
from libexperiment import generate_bootstrap_sample_with_outliers, generate_bootstrap_sample

import libexperiment

import numpy
import datetime
from datetime import timezone
import copy

from plot_log_likelihood_outlier_no_outlier import plot_log_likelihood_outlier_no_outlier


def main():
    
    rng = numpy.random.default_rng(12345)
    
    mean = 0.0
    stddev = 1.0
    
    n_outliers = 2
    number_of_samples_per_experiment = 100
    bootstrap_size = 100000
    
    optimized_parameter_data_1 = None
    optimized_parameter_data_2 = None
    optimized_parameters_1 = None
    optimized_parameters_2 = None
    
    try:
        
        optimized_parameter_data_1 = numpy.load(f'optimized_parameter_data_1_n{n_outliers}')
        optimized_parameter_data_2 = numpy.load(f'optimized_parameter_data_2_n{n_outliers}')
            
        optimized_parameters_1 = numpy.load(f'optimized_parameters_1_n{n_outliers}')
        optimized_parameters_2 = numpy.load(f'optimized_parameters_2_n{n_outliers}')
        
    except Exception as exception:
        print(f'failed to load data from file')
        
        data_1 = generate_experiment_data_gaussian(rng, number_of_samples_per_experiment, mean, stddev)
        data_2 = copy.deepcopy(data_1)
        data_2[0:n_outliers] = rng.normal(loc=0.0, scale=10*stddev, size=n_outliers)
        
        bootstrap_data_1 = generate_bootstrap_sample(rng, data_1, bootstrap_size)
        #bootstrap_data_2 = generate_bootstrap_sample_with_outliers(rng, data_2, n, n_outliers, stddev)
        bootstrap_data_2 = generate_bootstrap_sample(rng, data_2, bootstrap_size)
        
        # 1d -> 2d
        data_1 = numpy.reshape(data_1, (1, -1))
        data_2 = numpy.reshape(data_2, (1, -1))
            
        optimized_parameter_data_1 = libexperiment.optimize.optimize_gaussian_model(data_1, mean, stddev)
        data_1_log_likelihood = optimized_parameter_data_1[0][2]
            
        optimized_parameter_data_2 = libexperiment.optimize.optimize_gaussian_model(data_2, mean, stddev)
        data_2_log_likelihood = optimized_parameter_data_2[0][2]
        
        optimized_parameter_data_1 = numpy.array(optimized_parameter_data_1)
        optimized_parameter_data_2 = numpy.array(optimized_parameter_data_2)
        
        numpy.save(f'optimized_parameter_data_1_n{n_outliers}', optimized_parameter_data_1)
        numpy.save(f'optimized_parameter_data_2_n{n_outliers}', optimized_parameter_data_2)
        
        optimized_parameters_1 = \
            libexperiment.optimize.optimize_gaussian_model(bootstrap_data_1, mean, stddev)
            
        optimized_parameters_2 = \
            libexperiment.optimize.optimize_gaussian_model(bootstrap_data_2, mean, stddev)
            
        optimized_parameters_1 = numpy.array(optimized_parameters_1)
        optimized_parameters_2 = numpy.array(optimized_parameters_2)
            
        numpy.save(f'optimized_parameters_1_n{n_outliers}', optimized_parameters_1)   
        numpy.save(f'optimized_parameters_2_n{n_outliers}', optimized_parameters_2)
    
    
    plot_log_likelihood_outlier_no_outlier(
        optimized_parameters_1, optimized_parameters_2,
        data_1_log_likelihood, data_2_log_likelihood,
        filenames=[ f'log_likelihood_outlier_no_outlier_{n_outliers}.png', \
                    f'log_likelihood_outlier_no_outlier_{n_outliers}.pdf'])


def timed_main():

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':
    timed_main()
    