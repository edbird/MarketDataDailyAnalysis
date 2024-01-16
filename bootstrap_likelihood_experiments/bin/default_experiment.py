#!/usr/bin/env python3



from libexperiment import generate_experiment_data_gaussian, generate_bootstrap_sample
import libexperiment


import numpy
import datetime
from datetime import timezone



def main():
    
    rng = numpy.random.default_rng(12345)
    
    mean = 0.0
    stddev = 1.0
    
    data = generate_experiment_data_gaussian(rng, 100, mean, stddev)
    
    bootstraps = generate_bootstrap_sample(rng, data, 100000)

    # each of my original samples was drawn from the same distribution (Gaussian with
    # mean=0, stddev=1, so I would not expect a resampling to produce any different
    # results)
    
    # todo: perform the ML procedure, and maintain the parameters and LL value (
    # and success True/False flag)
    # flag up any failures...
    # probably just use the simplex method to give greater chance of success
    # ... plot the LL
    
    # try the same procedure where the original distribution is a uniform,
    # and where the fit distribution is uniform... maybe fitting a uniform is a
    # bad idea, because ML does not work well here... could use Poisson instead
    
    # 1d -> 2d
    data = numpy.reshape(data, (1, -1))
    
    optimized_parameters = libexperiment.optimize.optimize_gaussian_model(bootstraps, mean, stddev)
    optimized_parameter_data = libexperiment.optimize.optimize_gaussian_model(data, mean, stddev)
    data_log_likelihood = optimized_parameter_data[0][2]
    
    libexperiment.plot.plot_log_likelihood(optimized_parameters, data_log_likelihood,
                                           filenames=['log_likelihood.png', 'log_likelihood.pdf'])


def timed_main():

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':
    timed_main()
    