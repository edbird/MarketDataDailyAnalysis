#!/usr/bin/env python3

import matplotlib.pyplot as plt

import math
import pandas
import numpy
import scipy
import scipy.stats as stats

import multiprocessing

import datetime
from datetime import timezone
import os

#import plot_aapl_likelihood.plot_aapl_likelihood.optimize_module_binned as optimize_module_binned
import optimize_module_unbinned as optimize_module_unbinned

from experimental_database import ExperimentDatabase, ExperimentRecord

# this version does not separate generation of bootstrap data from
# fitting logic, each iteration should generate data then fit it
# need another version of this code which does the binned fitting
# procedure
# need to extend my database implementation to permit storage of
# multiple fitting methods, with binned vs unbinned, measured
# parameters, variable numbers of bins / bin widths


def main():

    # Load AAPL
    eod_aapl_us = pandas.read_csv('./market_data/eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    # Load NVDA
    eod_nvda_us = pandas.read_csv('./market_data/eod_nvda_us.csv', dtype='str', delimiter=',')
    eod_nvda_us['Date'] = pandas.to_datetime(eod_nvda_us['Date'])
    eod_nvda_us['Close'] = eod_nvda_us['Close'].apply(lambda close: float(close))

    # Create difference
    eod_aapl_us['Diff'] = eod_aapl_us['Close'].diff()
    aapl_us_diff_std = eod_aapl_us['Diff'].std()

    # Convert data from Pandas Dataframe to numpy array, drop NaN
    eod_aapl_us_diff = eod_aapl_us['Diff']
    eod_aapl_us_diff = eod_aapl_us_diff.dropna()
    eod_aapl_us_diff = eod_aapl_us_diff.to_numpy()

    # Input data is: eod_aapl_us_diff

    # load data from file
    # check if data exists
    # perform fit if not

    experimental_database = ExperimentDatabase()
    experimental_database.load('experiments_unbinned.exdb')

    if experimental_database.get_data() is None:

        experiment_record = do_work_data(eod_aapl_us_diff)
        experimental_database.add_experiment_record(experiment_record)

    number_of_existing_pseudodata_records = \
        len(experimental_database.get_pseudodata_all()) if experimental_database.get_pseudodata_all() is not None else 0

    # change this value to generate more data
    number_of_required_pseudodata_records = 1

    number_of_pseudodata_records_to_generate = \
        max(0, number_of_required_pseudodata_records - number_of_existing_pseudodata_records)

    for index in range(number_of_pseudodata_records_to_generate):

        index_offset = number_of_existing_pseudodata_records
        experiment_index = index + index_offset

        experiment_record = do_work_pseduodata(eod_aapl_us_diff, experiment_index)
        experimental_database.add_experiment_record(experiment_record)

    experimental_database.save('experiments_unbinned.exdb')


def generate_pseudodata(eod_aapl_us_diff):

    random_index = numpy.random.randint(len(eod_aapl_us_diff), size=len(eod_aapl_us_diff))
    bootstrap_sample = eod_aapl_us_diff[random_index]
    return bootstrap_sample


def do_work_pseduodata(eod_aapl_us_diff, experiment_index):

    pseudodata = generate_pseudodata(eod_aapl_us_diff)

    (mean, stddev, ll) = do_work(pseudodata)

    experiment_record = ExperimentRecord()
    experiment_record.set_record_type('pseudodata')
    experiment_record.set_experiment_index(experiment_index)
    experiment_record.set_dataset(eod_aapl_us_diff)
    experiment_record.set_measurements(amplitude=None, mean=mean, stddev=stddev, log_likelihood=ll)

    return experiment_record


def do_work_data(eod_aapl_us_diff):

    (mean, stddev, ll) = do_work(eod_aapl_us_diff)

    experiment_record = ExperimentRecord()
    experiment_record.set_record_type('data')
    experiment_record.set_dataset(eod_aapl_us_diff)
    experiment_record.set_measurements(amplitude=None, mean=mean, stddev=stddev, log_likelihood=ll)

    return experiment_record


def do_work(eod_aapl_us_diff):

    # Note: the argument is named `eod_aapl_us_diff` but it might be
    # pseudodata

    mean = eod_aapl_us_diff.mean()
    stddev = eod_aapl_us_diff.std()

    x0 = [mean, stddev]

    # TODO: exception handling
    optimize_result = optimize_module_unbinned.perform_fitting_procedure_unbinned(eod_aapl_us_diff, x0, 'data', True)

    ll_value_obs = -optimize_result['fun']

    print('--- solution ---')
    print(optimize_result)
    print(f'mean={mean}')
    print(f'stddev={stddev}')
    print(f'll={-optimize_module_unbinned.optimize_function_likelihood_unbinned(x0, eod_aapl_us_diff)}')
    mean_out = optimize_result["x"][0]
    stddev_out = optimize_result["x"][1]
    print(f'mean_out={mean_out}')
    print(f'stddev_out={stddev_out}')
    print(f'll={ll_value_obs}')

    return (mean_out, stddev_out, ll_value_obs)


if __name__ == '__main__':

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')
