

from pymongo import MongoClient

#import json

from libexperiment.experiment_record_mongo import ExperimentRecord



class MongoInterfaceException(Exception):

    def __init__(self, message):
        self.message = message

    def what(self):
        return self.message



def get_connection_client():

    connection_string = "mongodb://192.168.239:27017/market_data_analysis"
    connection = MongoClient(connection_string)
    return connection


def get_connection_database(connection):

    database = connection['market_data_analysis']
    return database


def send_experiment_record(connection, experiment_record):

    database = get_connection_database(connection)
    collection = None

    if experiment_record.experiment_type == 'data':
        collection = database['maximum_likelihood_data']

    elif experiment_record.experiment_type == 'pseudodata':
        collection = database['maximum_likelihood_pseudodata']

    else:
        raise MongoInterfaceException(f'invalid experiment type {experiment_record.experiment_type}')

    # insert record
    document = experiment_record.get_dictionary()
    #json_document = json.dumps(document)
    #collection.insert_one(json_document)
    collection.insert_one(document)


def check_if_data_experiment_result_exists(connection):

    database = get_connection_database(connection)
    collection = database['maximum_likelihood_data']

    document = collection.find()

    print(document)

    for document in document:
        print('found a document')

    return True


def get_experiment_id_offset(connection):

    database = get_connection_database(connection)
    collection = database['maximum_likelihood_pseudodata']

    document = collection.find_one(filter=None, sort={'experiment_id': -1})

    if document is None:
        return None
    else:
        return document['experiment_id']

    # TODO: test this and then accept stack overflow answer:
    # https://stackoverflow.com/questions/77518950/how-to-find-the-maximum-value-of-a-document-field-with-python-and-mongodb/77520349#77520349

    documents = collection.find().sort('experiment_id', -1).limit(1)

    if documents is None:
        print(f'get_experiment_id_offset: no documents, return None')
        return None
    else:
        for document in documents:
            return document['experiment_id']


def get_log_likelihood_values_optimize_success(connection, limit=None):

    database = get_connection_database(connection)
    collection = database['maximum_likelihood_pseudodata']

    query = {'optimize_success': True}
    documents = None

    if limit is not None:
        documents = collection.find(query, limit=limit)
    else:
        documents = collection.find(query)

    log_likelihood_values = []

    for document in documents:
        log_likelihood = document['optimize_log_likelihood']
        log_likelihood_values.append(log_likelihood)

    return log_likelihood_values


def get_log_likelihood_values_optimize_fail(connection, limit=None):

    database = get_connection_database(connection)
    collection = database['maximum_likelihood_pseudodata']

    query = {'optimize_success': False}
    documents = None

    if limit is not None:
        documents = collection.find(query, limit=limit)
    else:
        documents = collection.find(query)

    log_likelihood_values = []

    for document in documents:
        log_likelihood = document['optimize_log_likelihood']
        log_likelihood_values.append(log_likelihood)

    return log_likelihood_values


def get_experiment_record_optimize_fail(connection, skip=None):

    database = get_connection_database(connection)
    collection = database['maximum_likelihood_pseudodata']

    query = {'optimize_success': False}
    document = None

    if skip is not None:
        document = collection.find_one(query, skip=skip)
    else:
        document = collection.find_one(query)

    if document:
        return ExperimentRecord.from_mongo_document(experiment_type='pseudodata', mongo_document=document)
    else:
        return None


def get_experiment_records_optimize_fail(connection, limit=None):

    database = get_connection_database(connection)
    collection = database['maximum_likelihood_pseudodata']

    query = {'optimize_success': False}
    documents = None

    if limit is not None:
        documents = collection.find(query, limit=limit)
    else:
        documents = collection.find(query)

    experiment_records = []
    for document in documents:
        experiment_record = ExperimentRecord.from_mongo_document(experiment_type='pseudodata', mongo_document=document)
        experiment_records.append(experiment_record)

    return experiment_records

    #return ExperimentRecord.from_mongo_document(experiment_type='pseudodata', mongo_document=document)


def get_experiment_records(connection, limit=None):

    database = get_connection_database(connection)
    collection = database['maximum_likelihood_pseudodata']

    documents = None

    if limit is not None:
        documents = collection.find(limit=limit)
    else:
        documents = collection.find()

    experiment_records = []

    for document in documents:
        experiment_record = ExperimentRecord.from_mongo_document(experiment_type='pseudodata', mongo_document=document)
        experiment_records.append(experiment_record)

    return experiment_records

