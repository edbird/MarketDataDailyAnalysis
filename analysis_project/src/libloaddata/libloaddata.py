
import pandas


def load_diff_data_aapl():

    # Load AAPL
    #eod_aapl_us = pandas.read_csv('./market_data/eod_aapl_us.csv', dtype='str', delimiter=',')
    #eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    #eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))
    eod_aapl_us = load_data_aapl()

    # Create difference
    eod_aapl_us['Diff'] = eod_aapl_us['Close'].diff()

    # Convert data from Pandas Dataframe to numpy array, drop NaN
    eod_aapl_us_diff = eod_aapl_us['Diff']
    eod_aapl_us_diff = eod_aapl_us_diff.dropna()
    eod_aapl_us_diff = eod_aapl_us_diff.to_numpy()

    # Input data is: eod_aapl_us_diff
    return eod_aapl_us_diff


def load_data_aapl():

    # Load AAPL
    eod_aapl_us = pandas.read_csv('./market_data/eod_aapl_us.csv', dtype='str', delimiter=',')
    eod_aapl_us['Date'] = pandas.to_datetime(eod_aapl_us['Date'])
    eod_aapl_us['Close'] = eod_aapl_us['Close'].apply(lambda close: float(close))

    eod_aapl_us = eod_aapl_us[['Date', 'Close']]

    return eod_aapl_us


