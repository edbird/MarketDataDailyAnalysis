#!/usr/bin/env python3

from libexperiment import generate_experiment_data_gaussian
from libexperiment.plot import plot_log_likelihood_distribution

import libexperiment

import numpy
import datetime
from datetime import timezone


def main():
    
    rng = numpy.random.default_rng(12345)
    
    mean = 0.0
    stddev = 1.0
    
    number_of_samples_per_experiment = 100
    number_of_experiments = 1000
    
    optimized_parameters = None
    
    try:
        
        #optimized_parameter_data_1 = numpy.load(f'optimized_parameter_data_1_n{n_outliers}')
        #optimized_parameter_data_2 = numpy.load(f'optimized_parameter_data_2_n{n_outliers}')
            
        #optimized_parameters_1 = numpy.load(f'optimized_parameters_1_n{n_outliers}')
        #optimized_parameters_2 = numpy.load(f'optimized_parameters_2_n{n_outliers}')
        raise "not implemented"
        
    except Exception as exception:
        print(f'failed to load data from file')
        
        data_1 = []
        optimized_parameters = []
        
        for i in range(number_of_experiments):
            
            tmp_data_1 = generate_experiment_data_gaussian(rng, number_of_samples_per_experiment, mean, stddev)
            tmp_data_1 = numpy.reshape(tmp_data_1, (1, -1))
            data_1.append(tmp_data_1)
            
        data_1 = numpy.array(data_1)
        optimized_parameters = libexperiment.optimize.optimize_gaussian_model(data_1, mean, stddev)
            
        optimized_parameters = numpy.array(optimized_parameters)
        
        # 1d -> 2d
        #data_1 = numpy.reshape(data_1, (1, -1))
        #data_2 = numpy.reshape(data_2, (1, -1))
            
        #numpy.save(f'optimized_parameter_data_1_n{n_outliers}', optimized_parameter_data_1)
        #numpy.save(f'optimized_parameters_1_n{n_outliers}', optimized_parameters_1)   
    
    
    plot_log_likelihood_distribution(
        optimized_parameters,
        filenames=[ f'log_likelihood_value.png', \
                    f'log_likelihood_value.pdf'])


def timed_main():

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':
    timed_main()
    