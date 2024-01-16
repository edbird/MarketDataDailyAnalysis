

import numpy

import liboptimize

from libexperiment.experiment_record_mongo import ExperimentRecord


def generate_pseduodata(rng, data):

    #random_index = numpy.random.randint(len(data), size=len(data))
    random_index = rng.integers(len(data), size=len(data))
    bootstrap_sample = data[random_index]
    return bootstrap_sample


def run_pseudodata_experiment(rng, data, experiment_id):

    pseudodata = generate_pseduodata(rng, data)

    mean_init = pseudodata.mean()
    stddev_init = pseudodata.std()
    x_init = [mean_init, stddev_init]

    log_likelihood_init = -liboptimize.optimize_function_likelihood_unbinned(x_init, pseudodata)

    optimize_result = liboptimize.perform_fitting_procedure_unbinned(pseudodata, x_init)
    optimize_success = optimize_result.success
    optimize_message = optimize_result.message

    mean_opt = optimize_result.x[0]
    stddev_opt = optimize_result.x[1]
    x_opt = [mean_opt, stddev_opt]

    log_likelihood_opt = -optimize_result['fun']

    experiment_record = None

    if optimize_success:

        experiment_record = \
            ExperimentRecord.from_pseudodata(
                experiment_id=experiment_id,
                data=pseudodata,
                x_init=x_init,
                log_likelihood_init=log_likelihood_init,
                x_opt=x_opt,
                log_likelihood_opt=log_likelihood_opt,
                optimize_success=optimize_success,
                optimize_message=optimize_message)

    else:

        simplex_optimize_result = liboptimize.perform_fitting_procedure_unbinned_simplex(pseudodata, x_init)
        simplex_optimize_success = simplex_optimize_result.success
        simplex_optimize_message = simplex_optimize_result.message

        simplex_mean_opt = simplex_optimize_result.x[0]
        simplex_stddev_opt = simplex_optimize_result.x[1]
        simplex_x_opt = [simplex_mean_opt, simplex_stddev_opt]

        simplex_log_likelihood_opt = -simplex_optimize_result['fun']

        experiment_record = \
            ExperimentRecord.from_pseudodata_with_simplex(
                experiment_id=experiment_id,
                data=pseudodata,
                x_init=x_init,
                log_likelihood_init=log_likelihood_init,
                x_opt=x_opt,
                log_likelihood_opt=log_likelihood_opt,
                optimize_success=optimize_success,
                optimize_message=optimize_message,
                simplex_x_opt=simplex_x_opt,
                simplex_log_likelihood_opt=simplex_log_likelihood_opt,
                simplex_optimize_success=simplex_optimize_success,
                simplex_optimize_message=simplex_optimize_message)

    return experiment_record


def retry_run_pseudodata_experiment_get_optimize_result(pseudodata, experiment_id, method):

    mean_init = pseudodata.mean()
    stddev_init = pseudodata.std()
    x_init = [mean_init, stddev_init]

    log_likelihood_init = -liboptimize.optimize_function_likelihood_unbinned(x_init, pseudodata)

    optimize_result = None

    if method == 'simplex':
        optimize_result = liboptimize.perform_fitting_procedure_unbinned_simplex(pseudodata, x_init)
    elif method == 'default':
        optimize_result = liboptimize.perform_fitting_procedure_unbinned(pseudodata, x_init)
    else:
        raise 'failed, invalid argument for method'

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
        ExperimentRecord.from_pseudodata(
            experiment_id,
            pseudodata,
            x_init,
            log_likelihood_init,
            x_opt,
            log_likelihood_opt,
            optimize_success,
            optimize_message=optimize_message)

    return (experiment_record, optimize_result)

