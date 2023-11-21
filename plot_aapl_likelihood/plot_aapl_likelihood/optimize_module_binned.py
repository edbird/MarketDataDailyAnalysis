
import math
import numpy
import scipy
import scipy.stats as stats

from plot_aapl_likelihood_plotting_module import plot_failed_fit


class OptimizeFunctionException(Exception):

    def __init__(self, amplitude, mean, stddev):
        self.amplitude = amplitude
        self.mean = mean
        self.stddev = stddev


def optimize_function_likelihood(p, x, y):

    amplitude = p[0]
    mean = p[1]
    stddev = p[2]

    #print('optimize:')
    #print(amplitude)
    #print(mean)
    #print(stddev)
    #print(x)
    #print(y)

    # the change in price is expected to follow an exponential
    # distribution with a mean which is close to but not quite
    # zero
    normalization_constant = amplitude
    model = normalization_constant * stats.norm.pdf(x, mean, stddev)

    #fval = 1.0
    fval = 0.0

    # to create the likelihood function, for each bin midpoint (x)
    # get the value of the PDF, this gives the expected number of
    # events in this bin. Assume that the number of events in the
    # bin is distributed according to a Poisson distribution. The
    # total PDF for the whole histogram is a set of Poissons
    # multiplied together

    for index in range(len(model)):
        poisson_value = stats.poisson(model[index]).pmf(y[index])
        if poisson_value <= 0.0:
            print(f'\nindex: {index}, model: {model[index]}, y: {y[index]}, poisson: {poisson_value}')
            print(f'A={normalization_constant}, mean={mean}, stddev={stddev}')

            raise OptimizeFunctionException(amplitude, mean, stddev)

        log_poisson_value = math.log(poisson_value)

        #fval *= poisson_value
        fval += log_poisson_value
        #print(f'fval={fval}')

    # minimize the negative, gives a maximization
    return -fval


def perform_fitting_procedure(data, x0, index, disp=False):

    # Get histogram data
    (bin_contents, bin_edges) = \
        numpy.histogram(data, bins=20)

    # Create values of bin midpoints
    bin_midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    try:
        optimize_result = scipy.optimize.minimize(optimize_function_likelihood, x0, method='BFGS', \
            args=(bin_midpoints, bin_contents), \
            options={'disp': disp})

        return optimize_result

    except OptimizeFunctionException as exception:

        amplitude = exception.amplitude
        mean = exception.mean
        stddev = exception.stddev
        plot_failed_fit(index, bin_midpoints, bin_contents, amplitude, mean, stddev)

    except Exception as exception:
        print(f'failed fitting procedure: index={index}')
        print(f'{exception}')

    return None


