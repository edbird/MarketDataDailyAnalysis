

import numpy

import scipy.stats as stats



def optimize_function_least_squares(params, x, y, amplitude):

    mean = params[0]
    stddev = params[1]

    #print('optimize:')
    #print(amplitude)
    #print(mean)
    #print(stddev)
    #print(x)
    #print(y)

    model = amplitude * stats.norm.pdf(x, loc=mean, scale=stddev)
    error = numpy.sqrt(y)

    #residuals = (y - model) / error
    residuals = numpy.zeros(len(y))

    for index in range(len(residuals)):
        if error[index] <= 0.0:
            #print(f'warning: ignoring error[{index}] <= 0.0')
            pass
        else:
            value = (y[index] - model[index]) / error[index]
            residuals[index] = value

    return residuals

