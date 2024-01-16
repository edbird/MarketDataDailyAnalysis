#!/usr/bin/env python3

import libmongo

from libplot import plot_log_likelihood_distribution_2

import numpy


def main():

    # load data from MongoDB

    connection = libmongo.get_connection_client()
    
    log_likelihood_values = None
    
    try:
        log_likelihood_values = numpy.load('log_likelihood_any.npy')
        
    except Exception as exception:
        print(f'failed to load data from disk, loading from MongoDB')
        
        log_likelihood_values = \
            libmongo.get_log_likelihood_values(connection)
            
        log_likelihood_values = numpy.array(log_likelihood_values)
        
        numpy.save('log_likelihood_any.npy', log_likelihood_values)
        
        
    experiment_record = \
        libmongo.get_data_experiment_record(connection)
        
    data_log_likelihood_value = experiment_record.get_optimize_log_likelihood()

    plot_log_likelihood_distribution_2(data_log_likelihood_value,
                                     log_likelihood_values,
                                     filenames=['log_likelihood_distribution_any_2.png', 'log_likelihood_distribution_any_2.pdf'])
    

if __name__ == '__main__':

    main()

