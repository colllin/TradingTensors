import requests, re
import pandas as pd
class TypeBase():
    def __init__(self,dictonary):
        for key in dictonary :
            target = self._converter(dictonary[key])
            setattr(self, key, target)
    def _converter(self,target):
        if type(target) is str:
            if re.match("^\d+?\.\d+?$", target):
                target = float(target)
            elif re.match("^\d+$", target):
                target = int(target)
        return target
class Instrument(TypeBase):
    def __init__(self,dictonary):
        super().__init__(dictonary)
class Host():
    def __init__(self,rest,stream):
        self.rest = rest
        self.stream = stream
        for name in ['rest','stream']:
            setattr(self, name, getattr(self, name) + '/v3/')
class Request():
    def __init__(self,key,enviroment):
        if enviroment is 'sandbox':
            self.host = Host('http://api-sandbox.oanda.com','http://stream-sandbox.oanda.com' )
        elif enviroment is 'practice':
            self.host = Host('https://api-fxpractice.oanda.com','https://stream-fxpractice.oanda.com' )
        elif enviroment is 'real':
            self.host = Host('https://api-fxtrade.oanda.com','https://stream-fxtrade.oanda.com' )
        else :
            raise Exception('unknown enviroment')
        self.headers = {
            "Authorization" : 'Bearer ' + key,
            "X-Accept-Datetime-Format" : 'UNIX'
        }
    def get(self,url,params={}):
        response = requests.get(self.host.rest+url,headers=self.headers,params=params)
        json = response.json()
        err = None
        if 'errorMessage' in json:
            err = json['errorMessage']
            return err,None
        return None,json

class API():
    def __init__(self,key,enviroment,account_id=None):
        self.request = Request(key,enviroment)
        if account_id:
            self.account_id = account_id
            return;
        err,accounts = self.accounts()
        if err:
            raise Exception(err)
        using = None
        print('============== ACCOUNTS =================')
        for account in accounts:
            if using :
                print(account['id'])
            else:
                using = account['id']
                print(account['id'] + ' << using this account.')
        self.account_id = using
    def _get(self,url,name,clas=None,params={}):
        err, data = self.request.get(url,params)
        if data and name in data:
            data = data[name]
            if clas :
                instanced = []
                for single in data:
                    instanced.append(clas(single))
                data = instanced
        return err , data
    def accounts(self):
        return self._get('accounts','accounts')
    def instruments(self,account_id=None,instruments=None):
        if account_id is None:
            account_id = self.account_id
        params={}
        if instruments is not None:
            params['instruments'] = ','.join(instruments)
        return self._get(
            'accounts/' + account_id + '/instruments',
            'instruments',
            Instrument,
            params)

    def candles(self,params):
        instrument = params['instrument']
        params.pop('instrument',None)
        if 'count' not in params:
            params['count'] = 5000
        if 'alignmentTimezone' not in params:
            params['alignmentTimezone'] = 'Etc/GMT+0'
        if 'dailyAlignment' not in params:
            params['dailyAlignment'] = 0
        err, candles = self._get(
            'instruments/'+instrument+'/candles',
            'candles',
            params=params)
        if candles is None:
            return err,candles
        if len(candles) == 0:
            return 'no more data',None
        temp = []
        for candle in candles:
            if 'mid' in candle:
                for key, value in {'o':'open','h':'high','l':'low','c':'close'}.items():
                    candle[value] = candle['mid'][key]
                candle.pop('mid',None)
                #candle.pop('complete',None)
                temp.append(candle)
        candles = pd.DataFrame(temp)
        candles.time = pd.to_datetime(candles.time)
        candles.time = candles.time.dt.tz_localize('UTC')
        candles = candles.set_index('time')
        candles = candles.reindex_axis(['open','high','low','close','volume','complete'], axis=1)
        for name in ['open','high','low','close']:
            candles[name] = candles[name].astype('float')
        return err, candles
class OANDA():
    def __init__(self, key= None,enviroment='practice',account_id=None):
        if key is None:
            raise Exception('need api key')
        self.api = API(key,enviroment,account_id)