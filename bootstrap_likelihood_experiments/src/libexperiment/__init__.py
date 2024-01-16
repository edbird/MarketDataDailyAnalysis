
import numpy
import libexperiment.optimize as optimize
import libexperiment.plot as plot


def generate_experiment_data_gaussian(rng, n, mean=0.0, stddev=1.0):
    
    data = rng.normal(loc=mean, scale=stddev, size=n)
    
    return data


def generate_multiple_experiment_data_gaussian(
    rng, number_of_experiments, number_of_data_per_experiment, mean=0.0, stddev=1.0):
    
    data = rng.normal(loc=mean, scale=stddev, size=(number_of_experiments, number_of_data_per_experiment))
    
    return data
    
    
    
def generate_bootstrap_sample(rng, data, n):
    
    choices = rng.choice(data, size=(n, len(data)), replace=True)
    
    return choices


def generate_bootstrap_sample_with_outliers(rng, data, n, n_outliers, stddev):
    
    bootstrap_sample = generate_bootstrap_sample(rng, data, n)
    
    bootstrap_sample[:, 0:n_outliers] = rng.normal(loc=0.0, scale=10*stddev, size=(n, n_outliers))
    
    return bootstrap_sample
    
    

"""    
def generate_bootstrap_sample_with_outliers(rng, data, n_outliers, n):
    
    integers = rng.integers(low=0, high=len(data), size=len(data))
    
    bootstrap_data = data[integers]
    
    outlier_count = 0
    for value in integers:
        if value <= n_outliers:
            outlier_count += 1
            
    return (bootstrap_data, outlier_count)
"""

