#!/usr/bin/env python3

from libplot import plot_failed_fit
from libexperiment.pseudodata_experiment import retry_run_pseudodata_experiment_get_optimize_result
from libexperiment.experiment_record_mongo import ExperimentRecord

import numpy

import motor.motor_asyncio
import asyncio

async def async_driver():

    client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://192.168.0.239:27017')

    database = client['market_data_analysis']
    collection = database['experiment_record_aapl_pseudodata']

    document_count = 0

    query = {'optimize_success': False}
    async for document in collection.find(query):

        document_count += 1

        # convert document
        experiment_record = \
            ExperimentRecord.from_mongo_document(
                experiment_type='pseudodata',
                mongo_document=document)

        # process record
        process_experiment_record(experiment_record)

    print(f'total number of documents processed: {document_count}')


def process_experiment_record(experiment_record):

    experiment_id = experiment_record.get_experiment_id()
    pseudodata = experiment_record.get_dataset()

    # original optimization parameters
    optimize_mean = experiment_record.get_optimize_mean()
    optimize_stddev = experiment_record.get_optimize_stddev()

    # convert to numpy
    pseudodata = numpy.array(pseudodata)

    # re-perform the fit with the original method to find out what the message was
    (experiment_record_2, optimize_result) = \
        retry_run_pseudodata_experiment_get_optimize_result(pseudodata, experiment_id, 'default')

    # values should be the same
    optimize_mean_2 = experiment_record_2.get_optimize_mean()
    optimize_stddev_2 = experiment_record_2.get_optimize_stddev()

    if check_numerical_difference(optimize_mean, optimize_mean_2, 1.0e-6):
        print(f'optimize mean differs when re-running with same method')
        print(f'{optimize_mean}, {optimize_mean_2}')

    if check_numerical_difference(optimize_stddev, optimize_stddev_2, 1.0e-6):
        print(f'optimize stddev differs when re-running with same method')
        print(f'{optimize_stddev}, {optimize_stddev_2}')


    if optimize_result.success != True:
        message = optimize_result.message

        if message != 'Desired error not necessarily achieved due to precision loss.':
            print(f'found serious fit conversion failure, message={message}')
            print('optimize result')
            print(f'{optimize_result}')
            print(f'message={message}')
    elif optimize_result.success == True:
        print(f'error: somehow the fix managed to produce a successful result this time')
        print(f'this should never happen')


    # re-perform the fit with simplex method to check
    (experiment_record_3, optimize_result) = \
        retry_run_pseudodata_experiment_get_optimize_result(pseudodata, experiment_id, 'simplex')

    # check success
    if optimize_result.success != True:
        message = optimize_result.message

        print(f'fit failed with SIMPLEX method, this is a serious failure')
        print(f'message={message}')

    # check values compatiable
    optimize_mean_3 = experiment_record_3.get_optimize_mean()
    optimize_stddev_3 = experiment_record_3.get_optimize_stddev()

    if check_numerical_difference(optimize_mean, optimize_mean_3, 1.0e-2):
        print(f'optimize mean differs when re-running with SIMPLEX method')
        print(f'{experiment_record.get_experiment_id()}')
        print(f'{optimize_mean}, {optimize_mean_3}')
        print(f'percentage difference {calculate_percentage_difference(optimize_mean, optimize_mean_3)}')

    if check_numerical_difference(optimize_stddev, optimize_stddev_3, 1.0e-2):
        print(f'optimize stddev differs when re-running with SIMPLEX method')
        print(f'{experiment_record.get_experiment_id()}')
        print(f'{optimize_stddev}, {optimize_stddev_3}')
        print(f'percentage difference {calculate_percentage_difference(optimize_stddev, optimize_stddev_3)}')


def check_numerical_difference(x, y, percentage_threshold):

    difference = x - y
    abs_difference = abs(difference)
    abs_value = abs(x)
    fraction = abs_difference / abs_value
    percentage = fraction * 100.0

    if percentage >= percentage_threshold:
        return True
    else:
        return False


def calculate_percentage_difference(baseline, new_value):

    return 100.0 * abs((new_value - baseline) / baseline)


def main():
    asyncio.run(async_driver())


if __name__ == '__main__':
    main()
