

import scipy
import numpy



def maximum_likelihood_model_gaussian(params, data):
    
    mean = params[0]
    stddev = params[1]
    
    log_likelihood_values = scipy.stats.norm.logpdf(data, mean, stddev)
    fval = log_likelihood_values.sum()
    return -fval
    
    

def optimize_gaussian_model(input_data, init_mean, init_stddev):

    """
        `input_data` should be a 2d array
    """

    return_data = []
    
    for data in input_data:
        
        x_init = [init_mean, init_stddev]
        #data = bootstrap_data[bootstrap_data_index]
        
        x_init = [data.mean(), data.std()]

        optimize_result = scipy.optimize.minimize(
            maximum_likelihood_model_gaussian,
            x_init,
            method='Nelder-Mead',
            args=(data))
        
        if optimize_result.success != True:
            print(f'optimize failed: {optimize_result}')
            
        x_opt = optimize_result.x
        log_likelihood_opt = -optimize_result['fun']
        
        mean = x_opt[0]
        stddev = x_opt[1]
        
        next_data = (mean, stddev, log_likelihood_opt)
        
        return_data.append(next_data)
        
    return return_data
            
        
    
def optimize_gaussian_model_numpy(input_data):

    """
        `input_data` should be a 2d array
    """

    return_data = []
    
    for data in input_data:
        
        x_init = [data.mean(), data.std()]

        optimize_result = scipy.optimize.minimize(
            maximum_likelihood_model_gaussian,
            x_init,
            method='Nelder-Mead',
            args=(data))
        
        if optimize_result.success != True:
            print(f'optimize failed: {optimize_result}')
            
        x_opt = optimize_result.x
        log_likelihood_opt = -optimize_result['fun']
        
        mean = x_opt[0]
        stddev = x_opt[1]
        
        next_data = (mean, stddev, log_likelihood_opt)
        
        return_data.append(next_data)
        
    return_data = numpy.array(return_data)
        
    return return_data
            
        
    