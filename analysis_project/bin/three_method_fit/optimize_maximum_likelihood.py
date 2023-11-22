
import math
import numpy
import scipy.stats as stats


def optimize_function_maximum_likelihood_unbinned(params, data):

    mean = params[0]
    stddev = params[1]

    #print('optimize:')
    #print(mean)
    #print(stddev)
    #print(data)

    # the change in price is expected to follow an exponential
    # distribution with a mean which is close to but not quite
    # zero
    log_likelihood_values = stats.norm.logpdf(data, loc=mean, scale=stddev)

    # to create the likelihood function, add up the log of all
    # likelihood values
    fval = log_likelihood_values.sum()

    # minimize the negative, gives a maximization
    return -fval


def optimize_function_maximum_likelihood_binned(params, x, y, amplitude):

    mean = params[0]
    stddev = params[1]

    # the change in price is expected to follow an exponential
    # distribution with a mean which is close to but not quite
    # zero
    model = amplitude * stats.norm.pdf(x, loc=mean, scale=stddev)

    fval = 0.0

    # to create the likelihood function, for each bin midpoint (x)
    # get the value of the PDF, this gives the expected number of
    # events in this bin. Assume that the number of events in the
    # bin is distributed according to a Poisson distribution. The
    # total PDF for the whole histogram is a set of Poissons
    # multiplied together

    for index in range(len(model)):
        #poisson_value = stats.poisson(model[index]).pmf(y[index])
        log_poisson_value = stats.poisson(model[index]).logpmf(y[index])

        # log likelihood
        #fval += math.log(poisson_value)
        fval += log_poisson_value

    return -fval


