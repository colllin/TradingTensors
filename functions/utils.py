import logging


import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException
from talib import ATR, BBANDS

from settings.serverconfig import ID, TOKEN, HISTORICAL_DATA_LENGTH

DEFAULT_URL ={
    'practice': 'https://api-fxpractice.oanda.com',
    'live': 'https://api-fxtrade.oanda.com'
}

DEFAULT_HEADERS = {
    'Authorization' :'Bearer ' + TOKEN
}

############ OANDA REQUEST HANDLER ####################################
class OandaHandler(object):
    def __init__(self, granularity, mode='practice'):

        '''
        Input query parameters for retrieve instrument history Oanda API:
        '''
        self.granularity = granularity

        self.DEFAULT_URL = DEFAULT_URL[mode]
    def get_history(self, instrument):
        url = self.DEFAULT_URL + '/v3/instruments/{}/candles'.format(instrument)

        #Parameters required to retrieve history
        params = {
          "count": HISTORICAL_DATA_LENGTH,
          "granularity": self.granularity,
          "price": "M"
        }

        #Headers
        DEFAULT_HEADERS['Accept-Datetime-Format'] = 'RFC3339'

        #Make API call repeated
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, params=params).json()
            received = response["candles"]

        except (RequestException, KeyError) as e:
            print ("Failed to retrieve instrument history from Oanda, {}".format(e))
            return None


        #Store the history in a dictionary of list
        data = {
            "Open" : [],
            "High" : [],
            "Low" : [],
            "Close" : [],
            "Volume": [],
            "Date_Time" : [],
        }

        for r in received:
            data["Open"].append(float (r["mid"]["o"]) )
            data["High"].append(float (r["mid"]["h"]))
            data["Low"].append(float (r["mid"]["l"]))
            data["Close"].append(float (r["mid"]["c"]))
            data["Volume"].append(float(r["volume"]) )
            data["Date_Time"].append(r["time"])
        #Convert the dictionary to pandas DataFrame:
        #Date_Time (index) | Open | High | Low | Close | Volume
        df = pd.DataFrame(data)

        df["Date_Time"] = pd.to_datetime(df["Date_Time"])
        df.set_index(["Date_Time"], inplace=True)

        return df
    '''
    Get the Precision of the instrument
    '''
    def get_instrument_precision(self, instrument):

        url = self.DEFAULT_URL + '/v3/accounts/{}/instruments'.format(ID)

        params = {
            'instruments': instrument
        }

        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, params=params).json()
        except RequestException as e:
            print ("Error while retrieving instrument information: %s"%e)
            return None

        pip_base = response['instruments'][0]['pipLocation']

        return 10 ** pip_base