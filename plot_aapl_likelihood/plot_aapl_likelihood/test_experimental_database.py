
from experimental_database import ExperimentRecord
from experimental_database import ExperimentDatabase


def default_db_save_test():

    database = ExperimentDatabase()

    database.save('test_experimental_database.exdb')

    with open('test_experimental_database.exdb', 'r') as file:

        contents = file.read()
        print(contents)

        expected_contents = \
"""\
{
    "experiments": []
}\
"""

        #expected_contents = """{"experiments": []}"""

        assert contents == expected_contents, "default db save test failed"


def db_save_test():

    database = ExperimentDatabase()
    record = ExperimentRecord.from_pseudodata_experiment_index(10)

    database.add_experiment_record(record)

    filename = 'test_experimental_database_2.exdb'
    database.save(filename)

    database2 = ExperimentDatabase()
    database2.load(filename)

    record2 = database2.get_pseudodata_experiment_by_index(10)

    print(record)
    print(record2)

    assert record == record2, "db_save_test failed"


def main():

    default_db_save_test()
    db_save_test()


if __name__ == '__main__':
    main()

