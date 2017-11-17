import logging


import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException
from talib import ATR, BBANDS

from ..settings.serverconfig import ID, TOKEN

'''to access OANDA API'''


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


    def get_history(self, SYMBOL, HISTORY=5000):
        '''
        Retrieve history from Oanda
        return dataframe if success
        return None if failed
        '''
        QUERY_URL = self.DEFAULT_URL + '/v3/instruments/{}/candles'.format(SYMBOL)

        #Parameters required to retrieve history
        params = {
          "count": HISTORY,
          "granularity": self.granularity,
          "price": "M"
        }

        #Headers
        DEFAULT_HEADERS['Accept-Datetime-Format'] = 'RFC3339'

        #Make API call repeated
        try:
            response = requests.get(QUERY_URL, headers=DEFAULT_HEADERS, params=params).json()
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
    def get_instrument_precision(self, INSTRUMENT):

        QUERY_URL = self.DEFAULT_URL + '/v3/accounts/{}/instruments'.format(ID)

        params = {
            'instruments': INSTRUMENT
        }

        try:
            response = requests.get(QUERY_URL, headers=DEFAULT_HEADERS, params=params).json()
        except RequestException as e:
            print ("Error while retrieving instrument information: %s"%e)
            return None

        pip_base = response['instruments'][0]['pipLocation']

        return 10 ** pip_base


    def get_open_trades(self):

        QUERY_URL = self.DEFAULT_URL + '/v3/accounts/{}/openTrades'.format(ID)

        try:
            response = requests.get(QUERY_URL, headers=DEFAULT_HEADERS)
        except RequestException as e:
            print (e)
            return None

        return response.json()


    '''
    Open Position
    Return ENTRY_TIME and ENTRY_PRICE upon success
    Return None otherwise
    '''
    def open_position(self, INSTRUMENT, TYPE, UNITS=1):
        '''
        Open New Position
        Return time and entry_price upon success
        None, None if failed
        '''

        URL = self.DEFAULT_URL + '/v3/accounts/{}/orders'.format(ID)

        if TYPE == 'BUY':
            UNITS = UNITS * 1.0
        else:
            UNITS = UNITS * -1.0

        #Define payload for Order POSTING
        payload = {
            "order": {
                    "timeInForce": "FOK",
                    "units": UNITS,
                    "type": "MARKET",
                    "positionFill": "DEFAULT",
                    "instrument": INSTRUMENT
            }
        }
        #Make API request
        try:
            response = requests.post(URL, headers=DEFAULT_HEADERS, json=payload)
        except RequestException as e:
            print ("Failed to complete {} order for {} units of {}".format(TYPE, abs(UNITS), INSTRUMENT))
            return None, None, None


        #Request is not successful, return None
        if response.status_code != 201:
            print (
                "HTTP ERROR {}: Failed to complete {} order for {} units of {}".format(
                response.status_code, TYPE, abs(UNITS), INSTRUMENT)
                )
            return None, None, None


        #Extract the response json for time and price

        _json = response.json()
        try:
            transaction = _json["orderFillTransaction"]
        except KeyError as e:
            print (
                "Failed to Initiate order"
            )
            return None, None, None
        entry_price = transaction['price']
        entry_time = transaction['time']
        tradeID = transaction['id']

        ENTRY_TIME = pd.to_datetime(entry_time).strftime("%Y/%m/%d %H:%M")
        print("OANDA API: Successfully traded {} units of {} {}".format(UNITS, INSTRUMENT, ENTRY_TIME))
        return tradeID, entry_time, entry_price

    '''
    Close Positions
    '''
    def closeALLposition(self, INSTRUMENT, ORDER_TYPE):

        URL = self.DEFAULT_URL + '/v3/accounts/{}/positions/{}/close'.format(ID, INSTRUMENT)

        #Existing order is Long or Short
        if ORDER_TYPE == 'SELL':
            payload = {"shortUnits": "ALL"}
            _key = "shortOrderFillTransaction"
        else:
            payload = { "longUnits": "ALL"}
            _key = "longOrderFillTransaction"


        #Make API Request
        try:
            response = requests.put(URL, json=payload, headers=DEFAULT_HEADERS)
        except RequestException as e:
            print ('Failed to complete "PUT" request: {}'.format(e))
            return None, None, None

        if response.status_code != 200:
            print ("Incorrect status code")
            return None, None, None

        _json = response.json()
        exit_price = _json[_key]["price"]
        exit_time = _json[_key]["time"]
        pl = _json[_key]["pl"]

        EXIT_TIME = pd.to_datetime(exit_time).strftime("%Y/%m/%d %H:%M")
        print ("OANDA API: Successfully closed ALL positions in {} {}".format(INSTRUMENT, EXIT_TIME))

        return exit_time, exit_price, float(pl)


    def isTradeOpen(self, TRADE_ID):

        '''
        Check if Trade ID is still Open
        '''
        QUERY_URL = self.DEFAULT_URL + \
        "/v3/accounts/{accountID}/trades/{tradeSpecifier}"\
        .format(accountID=ID, tradeSpecifier=str(TRADE_ID))

        try:
            response = requests.get(QUERY_URL, headers=DEFAULT_HEADERS)
        except RequestException as e:
            print (e)
            return False

        response_json = response.json()

        try:
            open_trades = response_json['trade']
        except KeyError as e:
            print ("Trade Doesnt exist")
            return False

        isOpen = open_trades['state'] == 'OPEN'

        return isOpen



    def getOpenPL(self, TRADE_ID, SYMBOL):

        QUERY_URL = self.DEFAULT_URL + \
        "/v3/accounts/{accountID}/trades/{tradeSpecifier}"\
        .format(accountID=ID, tradeSpecifier=str(TRADE_ID))

        try:
            response = requests.get(QUERY_URL, headers=DEFAULT_HEADERS)
        except RequestException as e:
            print (e)
            return False

        response_json = response.json()
        try:
            trade= response_json['trade']
        except KeyError as e:
            print ("Trade Doesnt exist")

            result = {
            'Profit': None,
            'Exit Time': None,
            'Exit Price': None
        }

            return result
        #CHECK TO MAKE SURE IT IS THE RIGHT SYMBOL!

        return float(trade['unrealizedPL'])



    def getLatestTime(self, SYMBOL):

        QUERY_URL = self.DEFAULT_URL + '/v3/instruments/{}/candles'\
                    .format(SYMBOL)

        params = {
          "count": 1,
          "granularity": self.granularity,
          "price": "M"
        }

        DEFAULT_HEADERS['Accept-Datetime-Format'] = 'RFC3339'

        try:
            response = requests.get(
                QUERY_URL,
                headers=DEFAULT_HEADERS,
                params=params)

        except RequestException as e:
            print (e)
            return None

        responseJson = response.json()
        _time = responseJson["candles"][-1]["time"]

        return pd.to_datetime(_time).to_datetime()








