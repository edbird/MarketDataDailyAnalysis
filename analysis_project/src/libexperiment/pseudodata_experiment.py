

import numpy

import liboptimize

from libexperiment.experiment_record_mongo import ExperimentRecord


def generate_pseduodata(data):

    random_index = numpy.random.randint(len(data), size=len(data))
    bootstrap_sample = data[random_index]
    return bootstrap_sample


def run_pseudodata_experiment(data, experiment_id):

    pseudodata = generate_pseduodata(data)

    mean_init = pseudodata.mean()
    stddev_init = pseudodata.std()
    x_init = [mean_init, stddev_init]

    log_likelihood_init = liboptimize.optimize_function_likelihood_unbinned(x_init, pseudodata)

    optimize_result = liboptimize.perform_fitting_procedure_unbinned(pseudodata, x_init)
    optimize_success = optimize_result.success

    #if not optimize_result.success:
        # do something to re-try
        # print the result or otherwise save it?
        # save to the database?

    mean_opt = optimize_result.x[0]
    stddev_opt = optimize_result.x[1]
    x_opt = [mean_opt, stddev_opt]

    log_likelihood_opt = -optimize_result['fun']

    experiment_record = \
        ExperimentRecord.from_pseudodata(
            experiment_id,
            pseudodata,
            x_init,
            log_likelihood_init,
            x_opt,
            log_likelihood_opt,
            optimize_success)

    return experiment_record

