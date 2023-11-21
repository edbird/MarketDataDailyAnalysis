
import json
import os
import numpy


class ExperimentalDatabaseException(Exception):

    def __init__(self, what):
        self.what_string = what

    def what(self):
        return self.what_string



class ExperimentRecord():

    def __str__(self) -> str:
        return str(self.get_dictionary())

    def __eq__(self, __value: object) -> bool:

        dataset_check = False

        if __value.dataset is None and self.dataset is None:
            dataset_check = True
        elif __value.dataset is None or self.dataset is None:
            dataset_check = False
        else:
            dataset_check = len(__value.dataset) == len(self.dataset)

            if len(self.dataset) == 0:
                pass
            else:
                dataset_check = __value.dataset == self.dataset

        return \
            __value.experiment_index == self.experiment_index and \
            __value.record_type == self.record_type and \
            dataset_check and \
            __value.measurement_amplitude == self.measurement_amplitude and \
            __value.measurement_mean == self.measurement_mean and \
            __value.measurement_stddev == self.measurement_stddev and \
            __value.measurement_log_likelihood == self.measurement_log_likelihood

    def __init__(self) -> None:

        self.experiment_index = None
        self.record_type = None
        self.dataset = None
        self.measurement_amplitude = None
        self.measurement_mean = None
        self.measurement_stddev = None
        self.measurement_log_likelihood = None

    @classmethod
    def from_pseudodata_experiment_index(class_, experiment_index):
        tmp = class_()
        tmp.__from_pseudodata_experiment_index(experiment_index)
        return tmp

    @classmethod
    def from_data(class_):
        tmp = class_()
        tmp.__from_data()
        return tmp

    @classmethod
    def from_dictionary(class_, dictionary):
        tmp = class_()
        tmp.__from_dictionary(dictionary)
        return tmp

    def __from_pseudodata_experiment_index(self, experiment_index):

        self.experiment_index = experiment_index
        self.record_type = 'pseudodata'
        self.dataset = None
        self.measurement_amplitude = None
        self.measurement_mean = None
        self.measurement_stddev = None
        self.measurement_log_likelihood = None

    def __from_data(self):

        self.experiment_index = None
        self.record_type = 'data'
        self.dataset = None
        self.measurement_amplitude = None
        self.measurement_mean = None
        self.measurement_stddev = None
        self.measurement_log_likelihood = None

    def __from_dictionary(self, dictionary):

        dataset_list_string = dictionary['data']
        dataset = []
        if dataset_list_string is not None:
            dataset_list = json.loads(dataset_list_string)
            dataset = numpy.array(dataset_list)

        self.experiment_index = dictionary['index']
        self.record_type = dictionary['record_type']
        self.dataset = dataset
        self.measurement_amplitude = dictionary['amplitude']
        self.measurement_mean = dictionary['mean']
        self.measurement_stddev = dictionary['stddev']
        self.measurement_log_likelihood = dictionary['log_likelihood']

    def set_measurements(self, amplitude, mean, stddev, log_likelihood):

        self.measurement_amplitude = amplitude
        self.measurement_mean = mean
        self.measurement_stddev = stddev
        self.measurement_log_likelihood = log_likelihood

    def get_record_type(self):

        if self.record_type == 'data':
            return self.record_type
        elif self.record_type == 'pseudodata':
            return self.record_type
        else:
            raise ExperimentalDatabaseException(f'invalid record type {self.record_type}')

    def set_record_type(self, record_type_string):

        if record_type_string == 'data':
            self.record_type = record_type_string
        elif record_type_string == 'pseudodata':
            self.record_type = record_type_string
        else:
            raise ExperimentalDatabaseException(f'invalid record type {record_type_string}')

    def get_experiment_index(self):
        return self.experiment_index

    def set_experiment_index(self, experiment_index):
        self.experiment_index = experiment_index

    def set_dataset(self, dataset):

        self.dataset = dataset

    def get_dictionary(self):

        dataset_list = self.dataset.tolist() if self.dataset is not None else []
        dataset_list_string = json.dumps(dataset_list)

        dictionary = {
            'index': self.experiment_index,
            'record_type': self.record_type,
            'data': dataset_list_string,
            'amplitude': self.measurement_amplitude,
            'mean': self.measurement_mean,
            'stddev': self.measurement_stddev,
            'log_likelihood': self.measurement_log_likelihood,
        }

        return dictionary


class ExperimentDatabase():

    def __init__(self):
        self.dictionary = None
        self.__initialize_dictionary()

    def load(self, filename):

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                dictionary_string = file.read()
                dictionary = json.loads(dictionary_string)
                self.dictionary = dictionary

        else:
            pass

    def save(self, filename):

        dictionary_string = json.dumps(self.dictionary, indent=4)
        #dictionary_string = json.dumps(self.dictionary)

        with open(filename, 'w') as file:
            file.write(dictionary_string)

    def data_exists(self):
        if self.dictionary is None:
            return False
        else:
            if self.get_data() is None:
                return False
            else:
                return True

    def get_data(self):
        if self.dictionary is None:
            return None
        else:
            experiments = self.dictionary['experiments']
            for experiment in experiments:
                experiment_record = ExperimentRecord.from_dictionary(dictionary=experiment)
                if experiment_record.get_record_type() == 'data':
                    if experiment_record.get_experiment_index() is None:
                        return experiment_record
                    else:
                        raise ExperimentalDatabaseException(f'record type data has non-null experiment index')
                else:
                    continue

            return None

    def pseudodata_experiment_index_exists(self, experiment_index):
        if self.dictionary is None:
            return False
        else:
            if self.get_pseudodata_experiment_by_index(experiment_index) is None:
                return False
            else:
                return True

    def get_pseudodata_experiment_by_index(self, experiment_index):
        if self.dictionary is None:
            return None
        else:
            experiments = self.dictionary['experiments']
            for experiment in experiments:
                experiment_record = ExperimentRecord.from_dictionary(dictionary=experiment)
                if experiment_record.get_record_type() == 'data':
                    continue
                elif experiment_record.get_record_type() == 'pseudodata':
                    if experiment_record.get_experiment_index() == experiment_index:
                        return experiment_record
                else:
                    raise ExperimentalDatabaseException(f'invalid record type {experiment_record.get_record_type()}')

            return None

    def get_pseudodata_all(self):
        if self.dictionary is None:
            return None
        else:
            experiments = self.dictionary['experiments']
            experiment_records = []
            for experiment in experiments:
                experiment_record = ExperimentRecord.from_dictionary(dictionary=experiment)
                if experiment_record.get_record_type() == 'data':
                    continue
                elif experiment_record.get_record_type() == 'pseudodata':
                    experiment_records.append(experiment_record)
                else:
                    raise ExperimentalDatabaseException(f'invalid record type {experiment_record.get_record_type()}')

            return experiment_records

    def add_experiment_record(self, experiment_record):
        if self.dictionary is None:
            self.__initialize_dictionary()

        if experiment_record.get_record_type() == 'pseudodata':
            if self.pseudodata_experiment_index_exists(experiment_record.get_experiment_index()):
                raise ExperimentalDatabaseException(f'index {experiment_record.get_experiment_index()} exists')
        elif experiment_record.get_record_type() == 'data':
            if self.data_exists():
                raise ExperimentalDatabaseException(f'data exists')
        else:
            raise ExperimentalDatabaseException(f'invalid record type {experiment_record.get_record_type()}')

        experiment_dictionary = experiment_record.get_dictionary()
        self.dictionary['experiments'].append(experiment_dictionary)

    def __initialize_dictionary(self):
        self.dictionary = {
            'experiments': []
        }


def run_serialize_deserialize_test():

    experiment_record = ExperimentRecord.from_pseudodata_experiment_index(1)
    experiment_record.set_record_type('data')
    experiment_record.set_dataset(numpy.array([1, 2, 3]))
    experiment_record.set_measurements(amplitude=10.0, mean=11.0, stddev=12.0, log_likelihood=20.0)

    dictionary = experiment_record.get_dictionary()

    experiment_record_2 = ExperimentRecord.from_dictionary(dictionary)
    print(f'{experiment_record}')
    print(f'{experiment_record_2}')

    assert experiment_record == experiment_record_2, 'FAIL: run_serialize_deserialize_test'


def run_all_tests():
    run_serialize_deserialize_test()


if __name__ == '__main__':
    run_all_tests()

