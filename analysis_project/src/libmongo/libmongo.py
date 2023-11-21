

from pymongo import MongoClient

#import json



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

    document = collection.find_one(filter=None, sort={'experiment_id', -1})

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

