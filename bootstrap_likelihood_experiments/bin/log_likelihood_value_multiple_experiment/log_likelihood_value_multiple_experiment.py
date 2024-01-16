#!/usr/bin/env python3

from libexperiment import generate_multiple_experiment_data_gaussian

import libexperiment

import numpy
import datetime
from datetime import timezone

import matplotlib.pyplot as plt


def main():
    
    rng = numpy.random.default_rng(12345)
    
    mean = 0.0
    stddev = 1.0
    
    # variable
    #number_of_samples_per_experiment = 100
    number_of_samples_per_experiment = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    number_of_samples_per_experiment = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100]
    number_of_samples_per_experiment = numpy.array(number_of_samples_per_experiment)
    number_of_experiments = 10000
    
    optimized_parameters = None

    values_to_save = []
    
    try:
        
        #optimized_parameter_data_1 = numpy.load(f'optimized_parameter_data_1_n{n_outliers}')
        #optimized_parameter_data_2 = numpy.load(f'optimized_parameter_data_2_n{n_outliers}')
            
        #optimized_parameters_1 = numpy.load(f'optimized_parameters_1_n{n_outliers}')
        #optimized_parameters_2 = numpy.load(f'optimized_parameters_2_n{n_outliers}')
        raise "not implemented"
        
    except Exception as exception:
        print(f'failed to load data from file')
        
        figure = plt.figure()
        axis = figure.add_subplot(1, 1, 1)
        axis.set_xlabel('Bootstrap log-likelihood value')
        axis.set_ylabel('Number of pseudo-experiments')
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            
        for number_of_samples_per_experiment in number_of_samples_per_experiment:
            
            data = generate_multiple_experiment_data_gaussian(rng,
                        number_of_experiments=number_of_experiments,
                        number_of_data_per_experiment=number_of_samples_per_experiment)
            
            optimized_parameters = libexperiment.optimize.optimize_gaussian_model_numpy(data)
            log_likelihood_values = optimized_parameters[:,2]
            
            next_value_to_save = (number_of_samples_per_experiment, log_likelihood_values.mean(), log_likelihood_values.std())
            values_to_save.append(next_value_to_save)
            
            print(f'log_likelihood_values.mean() = {log_likelihood_values.mean()}')
            
            (ll_bin_counts, ll_bin_edges, _) = axis.hist(
                log_likelihood_values, bins=100, histtype='step')
        
        #numpy.save(f'optimized_parameter_data_1_n{n_outliers}', optimized_parameter_data_1)
        #numpy.save(f'optimized_parameters_1_n{n_outliers}', optimized_parameters_1)   
    
        filenames=[ f'log_likelihood_value_multiple.png', \
                    f'log_likelihood_value_multiple.pdf']
        
        for filename in filenames:
            figure.savefig(filename)
            
        values_to_save = numpy.array(values_to_save)
        numpy.save('values_to_save', values_to_save)
    
    #plot_log_likelihood_distribution(
    #    optimized_parameters,
    #    filenames=[ f'log_likelihood_value.png', \
    #                f'log_likelihood_value.pdf'])


def timed_main():

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':
    timed_main()
    