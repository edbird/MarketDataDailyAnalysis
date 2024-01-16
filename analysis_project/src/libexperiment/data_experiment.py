
import numpy

import liboptimize

from libexperiment.experiment_record_mongo import ExperimentRecord

import libmongo


def run_data_experiment(data):
    
    connection = libmongo.get_connection_client()
    data_experiment_result_exists = libmongo.check_if_data_experiment_result_exists(connection)
    connection.close()

    if data_experiment_result_exists:
        return None

    mean_init = data.mean()
    stddev_init = data.std()
    x_init = [mean_init, stddev_init]

    log_likelihood_init = liboptimize.optimize_function_likelihood_unbinned(x_init, data)

    optimize_result = liboptimize.perform_fitting_procedure_unbinned(data, x_init)
    optimize_success = optimize_result.success
    optimize_message = optimize_result.message

    #if not optimize_result.success:
        # do something to re-try
        # print the result or otherwise save it?
        # save to the database?

    mean_opt = optimize_result.x[0]
    stddev_opt = optimize_result.x[1]
    x_opt = [mean_opt, stddev_opt]

    log_likelihood_opt = -optimize_result['fun']

    experiment_record = \
        ExperimentRecord.from_data(
            data,
            x_init,
            log_likelihood_init,
            x_opt,
            log_likelihood_opt,
            optimize_success,
            optimize_message)

    return experiment_record


