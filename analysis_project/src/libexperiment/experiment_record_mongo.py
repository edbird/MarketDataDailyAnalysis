

import json
import os
import numpy


class ExperimentRecord():

    def __str__(self) -> str:
        return str(self.get_dictionary())


    def __init__(self) -> None:

        # Integer ID for experiment, only used for pseudodata.
        # None for real data
        self.experiment_id = None
        # 'data' or 'pseudodata'
        self.experiment_type = None
        # Ticker
        self.ticker = 'AAPL'
        # Experimental dataset
        self.dataset = None
        # Initial parameters
        self.init_mean = None
        self.init_stddev = None
        self.init_log_likelihood = None
        # Successful optimization?
        self.optimize_success = None
        self.optimize_message = None
        # Measured parameters
        self.optimize_mean = None
        self.optimize_stddev = None
        self.optimize_log_likelihood = None

        # Simplex
        # Successful optimization?
        self.simplex_optimize_success = None
        self.simplex_optimize_message = None
        # Measured parameters
        self.simplex_optimize_mean = None
        self.simplex_optimize_stddev = None
        self.simplex_optimize_log_likelihood = None


    @classmethod
    def from_data(class_, data, x_init, log_likelihood_init, x_opt, log_likelihood_opt, optimize_success, optimize_message):
        experiment_record = class_()

        experiment_record.experiment_id = None
        experiment_record.experiment_type = 'data'
        experiment_record.dataset = data
        experiment_record.init_mean = x_init[0]
        experiment_record.init_stddev = x_init[1]
        experiment_record.init_log_likelihood = log_likelihood_init

        experiment_record.optimize_success = optimize_success
        experiment_record.optimize_message = optimize_message
        experiment_record.optimize_mean = x_opt[0]
        experiment_record.optimize_stddev = x_opt[1]
        experiment_record.optimize_log_likelihood = log_likelihood_opt

        # TODO
        #experiment_record.simplex_optimize_success = simplex_optimize_success
        #experiment_record.simplex_optimize_message = simplex_optimize_message
        #experiment_record.simplex_optimize_mean = simplex_x_opt[0]
        #experiment_record.simplex_optimize_stddev = simplex_x_opt[1]
        #experiment_record.simplex_optimize_log_likelihood = simplex_log_likelihood_opt

        return experiment_record


    @classmethod
    def from_pseudodata(class_,
                        experiment_id, data,
                        x_init, log_likelihood_init,
                        x_opt, log_likelihood_opt, optimize_success, optimize_message):

        experiment_record = class_()

        experiment_record.experiment_id = experiment_id
        experiment_record.experiment_type = 'pseudodata'
        experiment_record.dataset = data
        experiment_record.init_mean = x_init[0]
        experiment_record.init_stddev = x_init[1]
        experiment_record.init_log_likelihood = log_likelihood_init

        # default optimization method
        experiment_record.optimize_success = optimize_success
        experiment_record.optimize_message = optimize_message
        experiment_record.optimize_mean = x_opt[0]
        experiment_record.optimize_stddev = x_opt[1]
        experiment_record.optimize_log_likelihood = log_likelihood_opt

        return experiment_record

    @classmethod
    def from_pseudodata_with_simplex(   class_,
                                        experiment_id, data,
                                        x_init, log_likelihood_init,
                                        x_opt, log_likelihood_opt, optimize_success, optimize_message,
                                        simplex_x_opt, simplex_log_likelihood_opt, simplex_optimize_success, simplex_optimize_message):

        experiment_record = class_()

        experiment_record.experiment_id = experiment_id
        experiment_record.experiment_type = 'pseudodata'
        experiment_record.dataset = data
        experiment_record.init_mean = x_init[0]
        experiment_record.init_stddev = x_init[1]
        experiment_record.init_log_likelihood = log_likelihood_init

        # default optimization method
        experiment_record.optimize_success = optimize_success
        experiment_record.optimize_message = optimize_message
        experiment_record.optimize_mean = x_opt[0]
        experiment_record.optimize_stddev = x_opt[1]
        experiment_record.optimize_log_likelihood = log_likelihood_opt

        # Simplex
        experiment_record.simplex_optimize_success = simplex_optimize_success
        experiment_record.simplex_optimize_message = simplex_optimize_message
        experiment_record.simplex_optimize_mean = simplex_x_opt[0]
        experiment_record.simplex_optimize_stddev = simplex_x_opt[1]
        experiment_record.simplex_optimize_log_likelihood = simplex_log_likelihood_opt

        return experiment_record


    @classmethod
    def from_mongo_document(class_, experiment_type, mongo_document):
        experiment_record = class_()

        experiment_record.experiment_id = mongo_document['experiment_id']
        experiment_record.experiment_type = experiment_type
        experiment_record.ticker = mongo_document['ticker']
        experiment_record.dataset = mongo_document['data']
        experiment_record.init_mean = mongo_document['init_mean']
        experiment_record.init_stddev = mongo_document['init_stddev']
        experiment_record.init_log_likelihood = mongo_document['init_log_likelihood']

        # default optimization method
        experiment_record.optimize_success = mongo_document['optimize_success']
        experiment_record.optimize_message = mongo_document['optimize_message']
        experiment_record.optimize_mean = mongo_document['optimize_mean']
        experiment_record.optimize_stddev = mongo_document['optimize_stddev']
        experiment_record.optimize_log_likelihood = mongo_document['optimize_log_likelihood']

        # Simplex
        experiment_record.simplex_optimize_success = mongo_document['simplex_optimize_success']
        experiment_record.simplex_optimize_message = mongo_document['simplex_optimize_message']
        experiment_record.simplex_optimize_mean = mongo_document['simplex_optimize_mean']
        experiment_record.simplex_optimize_stddev = mongo_document['simplex_optimize_stddev']
        experiment_record.simplex_optimize_log_likelihood = mongo_document['simplex_optimize_log_likelihood']

        return experiment_record


    def get_dictionary(self):

        dictionary = {
            'experiment_id': self.experiment_id,
            'experiment_type': self.experiment_type,
            'ticker': self.ticker,
            'data': self.dataset.tolist(),
            'init_mean': self.init_mean,
            'init_stddev': self.init_stddev,
            'init_log_likelihood': self.init_log_likelihood,
            # Default
            'optimize_success': self.optimize_success,
            'optimize_message': self.optimize_message,
            'optimize_mean': self.optimize_mean,
            'optimize_stddev': self.optimize_stddev,
            'optimize_log_likelihood': self.optimize_log_likelihood,
            # Simplex
            'simplex_optimize_success': self.simplex_optimize_success,
            'simplex_optimize_message': self.simplex_optimize_message,
            'simplex_optimize_mean': self.simplex_optimize_mean,
            'simplex_optimize_stddev': self.simplex_optimize_stddev,
            'simplex_optimize_log_likelihood': self.simplex_optimize_log_likelihood,
        }

        return dictionary


    def get_experiment_id(self):

        return self.experiment_id


    def get_ticker(self):
        return self.ticker

    def get_dataset(self):
        return self.dataset

    def get_init_mean(self):
        return self.init_mean

    def get_init_stddev(self):
        return self.init_stddev

    def get_init_log_likelihood(self):
        return self.init_log_likelihood

    def get_optimize_success(self):
        return self.optimize_success

    def get_optimize_message(self):
        return self.optimize_message

    def get_optimize_mean(self):
        return self.optimize_mean

    def get_optimize_stddev(self):
        return self.optimize_stddev

    def get_optimize_log_likelihood(self):
        return self.optimize_log_likelihood

    def get_simplex_optimize_success(self):
        return self.simplex_optimize_success

    def get_simplex_optimize_message(self):
        return self.simplex_optimize_message

    def get_simplex_optimize_mean(self):
        return self.simplex_optimize_mean

    def get_simplex_optimize_stddev(self):
        return self.simplex_optimize_stddev

    def get_simplex_optimize_log_likelihood(self):
        return self.simplex_optimize_log_likelihood

