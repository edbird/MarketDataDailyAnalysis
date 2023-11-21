
import scipy
import scipy.stats as stats


def optimize_function_likelihood_unbinned(params, data):

    mean = params[0]
    stddev = params[1]

    # the change in price is expected to follow an exponential
    # distribution with a mean which is close to but not quite
    # zero
    log_likelihood_values = stats.norm.logpdf(data, mean, stddev)

    # to create the likelihood function, add up the log of all
    # likelihood values
    fval = log_likelihood_values.sum()

    # minimize the negative, gives a maximization
    return -fval


def perform_fitting_procedure_unbinned(data, x0, disp=False):

    optimize_result = \
        scipy.optimize.minimize(
            optimize_function_likelihood_unbinned,
            x0,
            method='BFGS',
            args=(data),
            options={'disp': disp}
        )

    return optimize_result

