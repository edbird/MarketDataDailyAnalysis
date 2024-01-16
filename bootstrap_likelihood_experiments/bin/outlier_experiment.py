#!/usr/bin/env python3



from libexperiment import generate_experiment_data_gaussian, generate_bootstrap_sample
from libexperiment import generate_bootstrap_sample_with_outliers
import libexperiment


import numpy
import datetime
from datetime import timezone



def main():
    
    rng = numpy.random.default_rng(12345)
    
    mean = 0.0
    stddev = 1.0
    
    data = generate_experiment_data_gaussian(rng, 100, mean, stddev)
    
    #n_outliers = 1
    #random_index = rng.integers(low=0, high=len(data), size=n_outliers)
    #data[random_index] = rng.normal(loc=0.0, scale=10*stddev, size=n_outliers)
    #print(f'{data[0:n_outliers+1]}')
    #data[0:n_outliers] = rng.normal(loc=0.0, scale=10*stddev, size=n_outliers)
    #print(f'{data[0:n_outliers+1]}')
    
    #bootstraps = generate_bootstrap_sample(rng, data, 100000)
    #(bootstraps, outlier_count) = generate_bootstrap_sample_with_outliers(rng, data, n_outliers, 100000)
    
    #bootstrap_lists = {}
    #for (index, bootstrap_dataset) in enumerate(bootstraps):
    #    
    #    n_outliers = outlier_count[index]
    #    
    #    if n_outliers in bootstrap_lists:
    #        bootstrap_lists[n_outliers].append(bootstrap_dataset)
    #    else:
    #        bootstrap_lists[n_outliers] = [bootstrap_dataset]
     
    bootstrap_data_list = []
    n_outliers_list = [0, 1, 2, 5, 10, 20, 50, 60, 80, 90, 95, 100]
    
    for n_outliers in n_outliers_list:
        print(f'generating data for n_outliers={n_outliers}')
        bootstrap_data = generate_bootstrap_sample_with_outliers(rng, data, 100000, n_outliers, stddev)
        bootstrap_data_list.append(bootstrap_data)
    
    # 1d -> 2d
    data = numpy.reshape(data, (1, -1))
    
    optimized_parameters_list = []
    for index, bootstrap_data in enumerate(bootstrap_data_list):
        n_outliers = n_outliers_list[index]
        print(f'optimizing data for n_outliers={n_outliers}')
        optimized_parameters = libexperiment.optimize.optimize_gaussian_model(bootstrap_data, mean, stddev)
        optimized_parameters_list.append(optimized_parameters)
        
    optimized_parameter_data = libexperiment.optimize.optimize_gaussian_model(data, mean, stddev)
    data_log_likelihood = optimized_parameter_data[0][2]
    
    n_outliers = 100
    libexperiment.plot.plot_log_likelihood_list(optimized_parameters_list, data_log_likelihood,
        filenames=[f'log_likelihood_outlier_{n_outliers}.png', f'log_likelihood_outlier_{n_outliers}.pdf'],
        labels=['outliers=0', '1', '2', '5', '10', '20', '50', '60', '80', '90', '95', '100'])


def timed_main():

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':
    timed_main()
    