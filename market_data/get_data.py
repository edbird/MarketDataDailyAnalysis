#!/usr/bin/env python3


import requests


def main():

    #url = f'https://eodhd.com/api/eod/AAPL.US?'
    url = f'https://eodhd.com/api/eod/NVDA.US?'

    params = {
        'api_token': '',
        'fmt': 'csv',
        #'from':
        #'to':
    }

    response = requests.get(url, params)

    with open('output.csv', 'w') as f:

        f.write(response.text)




if __name__ == '__main__':
    main()
