

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
        # Experimental dataset
        self.dataset = None
        # Initial parameters
        self.init_mean = None
        self.init_stddev = None
        self.init_log_likelihood = None
        # Successful optimization?
        self.optimize_success = None
        # Measured parameters
        self.optimize_mean = None
        self.optimize_stddev = None
        self.optimize_log_likelihood = None


    @classmethod
    def from_data(class_, data, x_init, log_likelihood_init, x_opt, log_likelihood_opt, optimize_success):
        experiment_record = class_()

        experiment_record.experiment_id = None
        experiment_record.experiment_type = 'data'
        experiment_record.dataset = data
        experiment_record.init_mean = x_init[0]
        experiment_record.init_stddev = x_init[1]
        experiment_record.init_log_likelihood = log_likelihood_init
        experiment_record.optimize_success = optimize_success
        experiment_record.optimize_mean = x_opt[0]
        experiment_record.optimize_stddev = x_opt[1]
        experiment_record.optimize_log_likelihood = log_likelihood_opt

        return experiment_record


    @classmethod
    def from_pseudodata(class_, experiment_id, data, x_init, log_likelihood_init, x_opt, log_likelihood_opt, optimize_success):
        experiment_record = class_()

        experiment_record.experiment_id = experiment_id
        experiment_record.experiment_type = 'pseudodata'
        experiment_record.dataset = data
        experiment_record.init_mean = x_init[0]
        experiment_record.init_stddev = x_init[1]
        experiment_record.init_log_likelihood = log_likelihood_init
        experiment_record.optimize_success = optimize_success
        experiment_record.optimize_mean = x_opt[0]
        experiment_record.optimize_stddev = x_opt[1]
        experiment_record.optimize_log_likelihood = log_likelihood_opt

        return experiment_record


    def get_dictionary(self):

        dictionary = {
            'experiment_id': self.experiment_id,
            'experiment_type': self.experiment_type,
            'data': self.dataset.tolist(),
            'init_mean': self.init_mean,
            'init_stddev': self.init_stddev,
            'init_log_likelihood': self.init_log_likelihood,
            'optimize_success': self.optimize_success,
            'optimize_mean': self.optimize_mean,
            'optimize_stddev': self.optimize_stddev,
            'optimize_log_likelihood': self.optimize_log_likelihood,
        }

        return dictionary


    def get_experiment_id(self):

        return self.experiment_id


