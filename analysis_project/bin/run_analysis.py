#!/usr/bin/env python3

from libloaddata import load_diff_data_aapl

from libexception.exception_types import AnalysisException

from libexperiment.data_experiment import run_data_experiment
from libexperiment.pseudodata_experiment import run_pseudodata_experiment

import libmongo

import argparse
import os
import sys
import datetime
from datetime import timezone

from multiprocessing import Pool


def main():

    argument_parser = argparse.ArgumentParser(description='Run Experiment Data Generation')
    argument_parser.add_argument('--mode', metavar='mode', type=str, help='mode: [data|pseudodata]', required=True)
    argument_parser.add_argument('--num-workers', metavar='num_workers', type=int, help='number of workers [n]', required=False)
    argument_parser.add_argument('--num-pseudodata-experiments-per-worker', metavar='num_pseudodata_experiments_per_worker', type=int, help='number of pseudodata experiments per worker [n]', required=False)
    args = argument_parser.parse_args()

    # load data
    eod_aapl_us = load_diff_data_aapl()

    if args.mode == 'data':

        if args.num_workers is not None:
            raise AnalysisException(f'invalid argument combination, --num-workers is not valid in mode {args.mode}')

        # run data analysis
        # save to MongoDB

        experiment_record = run_data_experiment(eod_aapl_us)

        if experiment_record is None:
            print(f'data result already exists in database')

        else:
            connection = libmongo.get_connection_client()
            libmongo.send_experiment_record(connection, experiment_record)
            connection.close()


    elif args.mode == 'pseudodata':

        num_workers = args.num_workers
        num_pseudodata_experiments_per_worker = args.num_pseudodata_experiments_per_worker

        if num_workers is None:
            raise AnalysisException(f'argument --num-workers is required in mode {args.mode}')

        if num_pseudodata_experiments_per_worker is None:
            raise AnalysisException(f'argument --number-of-pseudodata-experiments-per-worker is required in mode {args.mode}')

        # launch worker processes
        # collect results
        # apply id to results ???
        # save to MongoDB

        connection = libmongo.get_connection_client()
        experiment_id_offset = libmongo.get_experiment_id_offset(connection)
        connection.close()

        if experiment_id_offset is None:
            experiment_id_offset = 0
        else:
            experiment_id_offset = experiment_id_offset + 1

        # send data to MongoDB after generating this number of results
        flush_batch_size = 10000

        #experiment_records = \
        run_pseudodata_experiment_parallel_driver(
            eod_aapl_us,
            experiment_id_offset,
            num_workers,
            num_pseudodata_experiments_per_worker,
            flush_batch_size)

    else:
        raise AnalysisException(f'invalid argument for mode {args.mode}')


def run_pseudodata_experiment_parallel_driver(data, experiment_id_offset, num_workers, batch_size, flush_batch_size):

    args_list = [
        (data, experiment_id_offset, worker_index, num_workers, batch_size, flush_batch_size)
            for worker_index in range(num_workers)
    ]

    print(f'Dispatching {num_workers} worker processes')

    with Pool(processes=num_workers) as pool:

        pool.map(parallel_function, args_list)
        #experiment_record_lists = pool.map(parallel_function, args_list)

        #experiment_records = flatten_list(experiment_record_lists)

        # check data ordering

        #last_experiment_id = None
        #for experiment_record in experiment_records:
        #    experiment_id = experiment_record.get_experiment_id()
        #
        #    if last_experiment_id is not None:
        #        if experiment_id < last_experiment_id:
        #            print(f'out of order data')
        #
        #    last_experiment_id = experiment_id

        # make sorted (no longer needed, data is sorted)

        #experiment_records = sorted(experiment_records, key = lambda experiment_record: experiment_record.get_experiment_id())

        #return experiment_records


def flatten_list(list):

    return [item for each_list in list for item in each_list]


def parallel_function(args):

    # expand arguments
    (data, experiment_id_offset, worker_index, num_workers, batch_size, flush_batch_size) = args

    print(f'Starting worker {worker_index}')

    experiment_records = []

    experiment_batch_id_offset = worker_index * batch_size
    print(f'experiment_batch_id_offset={experiment_batch_id_offset}')

    flush_batch_index_counter = 0

    for batch_index in range(batch_size):

        #experiment_id = experiment_id_offset + batch_index * num_workers + worker_index
        experiment_id = experiment_id_offset + experiment_batch_id_offset + batch_index

        experiment_record = run_pseudodata_experiment(data, experiment_id)
        #experiment_record.experiment_id = experiment_id

        experiment_records.append(experiment_record)

        # flush data on final iteration, or when flush_batch_size is reached
        # the greater than accounts for `flush_batch_size` being negative or zero
        if (flush_batch_index_counter + 1 >= flush_batch_size) or (batch_index + 1 >= batch_size):

            print(f'Worker {worker_index} flush results')

            # send data to MongoDB
            connection = libmongo.get_connection_client()
            for experiment_record in experiment_records:
                libmongo.send_experiment_record(connection, experiment_record)

            connection.close()

            experiment_records.clear()
            flush_batch_index_counter = 0

        else:
            flush_batch_index_counter += 1

    #return experiment_records


def timed_main():

    time_start = datetime.datetime.now(timezone.utc)

    main()

    time_end = datetime.datetime.now(timezone.utc)
    time_diff = time_end - time_start
    print(f'runtime: {time_diff}')


if __name__ == '__main__':

    timed_main()