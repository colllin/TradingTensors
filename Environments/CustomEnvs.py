import numpy as np
import pandas as pd
from settings.serverconfig import HISTORICAL_DATA_LENGTH, TRAIN_SPLIT, ID, TOKEN
from functions.utils import OandaHandler
from Fx import OANDA
from sklearn.model_selection import train_test_split

class Observation():
    def __init__(self,
                 key,
                 account_id,
                 enviroment='practice'):
        self.oanda = OANDA(key=key,account_id=account_id,enviroment='practice')
    def create(self,
               instrument,
               granularity,
               count,
               lookback_period):
        _ ,self.instrument = self.oanda.api.instruments(instruments=[instrument])
        self.instrument = self.instrument[0]
        params = {
            'instrument' : instrument,
            'granularity' : granularity,
            "count" : count
        };
        _, candles = self.oanda.api.candles(params)
        states = pd.DataFrame(index=candles.index)
        states['open'] = candles['open'].pct_change()
        # 平均を引いて 標準偏差で割る
        # http://data-science.gr.jp/theory/tbs_standardization.html
        states.open = (states.open - states.open.mean()) / states.open.std()

        if lookback_period > 0:
            original = states.copy()
            for i in range(0,lookback_period):
                shifted = original.shift(i+1)
                states = states.join(shifted, rsuffix="_t-{}".format(i+1))
        states.dropna(axis=0, how='any', inplace=True)
        self.states = states
        '''States Normalization ここでやるのと微妙に値が違う
        # 平均を引いて 標準偏差で割る
        # http://data-science.gr.jp/theory/tbs_standardization.html
        self.states = (states.values - np.mean(states.values, 0)) / np.std(states.values, 0)'''

        candles = candles.loc[states.index.tolist(), :]
        candles['reward'] = ( candles.close - candles.open ) / (10 ** self.instrument.pipLocation)

        self.candles = candles
class OandaEnv():
    def __init__(self,
                 instrument,
                 granularity,
                 training=True,
                 trade_duration=1,
                 lookback_period=0,
                 mode='practice'):
        self.api_Handle = OandaHandler(granularity, mode)

        self.instrument = instrument
        # Attributes to create state space
        self.lookback_period = lookback_period  # how many periods to lookback

        # Pull primary symbol from Oanda API
        primary_data = self.api_Handle.get_history(self.instrument,HISTORICAL_DATA_LENGTH)
        assert primary_data is not None, "primary_data is not DataFrame"

        states_df = pd.DataFrame(index=primary_data.index)

        states_df['Returns'] = primary_data['Open'].pct_change()

        # Shift Data if there are any lookback period
        if self.lookback_period > 0:
            original = states_df.copy()
            for i in range(0,self.lookback_period):
                _shifted = original.shift(i+1)
                states_df = states_df.join(_shifted, rsuffix="_t-{}".format(i+1))

        states_df.dropna(axis=0, how='any', inplace=True)

        self.data = primary_data.loc[states_df.index.tolist(), :]

        '''States Normalization'''
        self.states = (states_df.values - np.mean(states_df.values, 0)) / np.std(states_df.values, 0)

        #To be used in every step of Simulator
        self.Open = self.data.Open.values
        self.Dates = self.data.index.to_pydatetime().tolist()

        #Reward: (CLOSE - open) / (0.0001)
        precision = self.api_Handle.get_instrument_precision(instrument)
        self.reward_pips = (self.data['Close'] - self.data['Open']).values / precision

        '''
        Define the first and last index of states during training
        '''
        '''Before executing, check that states is defined'''
        assert self.states is not None, "No state space!"
        data_count = self.states.shape[0]
        '''Define boundary index for training and testing'''
        self.train_end_idx = int(TRAIN_SPLIT*data_count)
        self.test_start_idx = self.train_end_idx + 1
        self.test_end_idx = data_count - 1

        currencies = instrument.split('_')
        self.portfolio = Portfolio2(
            currencies=currencies,
            trade_duration=trade_duration,
            instrument=instrument)

        self.reset(training) #Reset to initialize curr_idx and end_idx

    def actions(self):
        """
        The actions are aligned with the available currencies that we can hold.
        One model must hold all of its capital in a single currency.
        To diversify your capital, run multiple bots, and consider training them individually.
        """
        actions = {}
        for c,i in enumerate(self.currencies):
            actions[c] = i

        return actions
        # return range(len(self.currencies))

    def step(self, action):
        reward = self.reward_pips[self.curr_idx] #Current Reward: Current Close - Close Open
        THIS_OPEN = self.Open[self.curr_idx] #Current Open
        THIS_TIME = self.Dates[self.curr_idx]
        self.curr_idx += 1
        done = self.curr_idx >= self.end_idx
        new_obs = self.states[self.curr_idx] #Next State

        action, reward = self.portfolio.newCandleHandler(
            action=action, time=THIS_TIME,
            open=THIS_OPEN, reward=reward)

        return new_obs, action, reward, done

    def reset(self, training):
        self.training = training
        self.portfolio.reset()
        if self.training:
            self.curr_idx = 0
            self.end_idx = self.train_end_idx
        else:
            self.curr_idx = self.test_start_idx
            self.end_idx = self.test_end_idx
        '''Edge Case: Step function will cross boundary of data '''
        if self.curr_idx == self.end_idx:
            raise Exception("Please use more history!")

        #Return the first instance of the state space
        return self.states[self.curr_idx]
class Trade():
    def __init__(self,
            instrument,
            id,
            type,
            time,
            open):
        self.id = id
        self.entry_time = time
        self.exit_time = 0
        self.entry_price = open
        self.exit_price = 0
        self.profit = 0
        self.duration = 0
        self.type = type
        self.instrument = instrument

class Portfolio():
    '''
    Portfolio imposes a fixed-duration trading regime
    No StopLoss is required, trades are closed automatically once they reach the specified duration
    '''
    def __init__(self,trade_duration,instrument):

        self.trade_duration = trade_duration
        self.instrument = instrument
        self.reset()

    def reset(self):

        #Cumulative reward in this run (in pips)
        self.total_reward = 0

        #Cumulative trades in this run
        self.total_trades = 0

        #History of cumulative reward
        self.equity_curve = [] #TO BE OUTSOURCED TO AGENT

        #Trade Profile
        self.trade = None
        self.trades = [] #Collection of trades

    def newCandleHandler(self, action,time,open,reward):
        if self.trade is not None:
            #Increase trade duration
            self.trade.duration += 1

            #Check if duration limit is reached
            if self.trade.duration >= self.trade_duration:
                #Close Trade
                self.closeTrade(time,open)
            else:
                #Continue Holding
                reward = self._updateProperties(reward)
                #Reset the action 2
                return 2, reward

        if action == 2:
            # Do Nothing
            self.equity_curve.append(self.total_reward)
            return action, 0
        #TAKE A TRADE
        return self.openTrade(action, time,open,reward)


    def openTrade(self, action, time,open,reward):
        self.total_trades += 1
        #Train/Test Mode
        type = 'BUY' if action == 0 else 'SELL'
        self.trade = Trade(
            self.instrument,
            self.total_trades,
            type,
            time,
            open)

        reward = self._updateProperties(reward)

        return action, reward
    def _updateProperties(self,reward):
        #Manipulate reward
        multiplier = 1.0 if self.trade.type == 'BUY' else -1.0
        reward = reward * multiplier


        #Accumulate reward
        self.trade.profit += reward
        self.total_reward += reward

        #Update Equity
        self.equity_curve.append(self.total_reward)
        return reward

    def closeTrade(self, time,open):
        #Close the trade in Train/Test Mode
        self.trade.exit_time = time
        self.trade.exit_price = open
        self.trades.append(self.trade)
        self.trade = None

class Portfolio2():
    '''
    Portfolio2 tracks account balances in primary and alternative currency, and a "net worth" value in the primary currency.
    FIXME: Consider StopLoss!!!
    No StopLoss is required, trades are closed automatically once they reach the specified duration
    '''
    def __init__(self,trade_duration,instrument):

        self.trade_duration = trade_duration
        self.instrument = instrument
        self.reset()

    def reset(self):

        #Cumulative reward in this run (in pips)
        self.total_reward = 0

        #Cumulative trades in this run
        self.total_trades = 0

        #History of cumulative reward
        self.equity_curve = [] #TO BE OUTSOURCED TO AGENT

        #Trade Profile
        self.trade = None
        self.trades = [] #Collection of trades

    def newCandleHandler(self, action,time,open,reward):
        # Close open trades if they reach the max trade duration
        if self.trade is not None:
            #Increase trade duration
            self.trade.duration += 1

            #Check if duration limit is reached
            if self.trade.duration >= self.trade_duration:
                #Close Trade
                self.closeTrade(time,open)
            else:
                #Continue Holding
                reward = self._updateProperties(reward)
                #Reset the action 2
                return 2, reward

        if action == 2:
            # Do Nothing
            self.equity_curve.append(self.total_reward)
            return action, 0
        #TAKE A TRADE
        return self.openTrade(action, time,open,reward)


    def openTrade(self, action, time,open,reward):
        self.total_trades += 1
        #Train/Test Mode
        type = 'BUY' if action == 0 else 'SELL'
        self.trade = Trade(
            self.instrument,
            self.total_trades,
            type,
            time,
            open)

        reward = self._updateProperties(reward)

        return action, reward
    def _updateProperties(self,reward):
        #Manipulate reward
        multiplier = 1.0 if self.trade.type == 'BUY' else -1.0
        reward = reward * multiplier


        #Accumulate reward
        self.trade.profit += reward
        self.total_reward += reward

        #Update Equity
        self.equity_curve.append(self.total_reward)
        return reward

    def closeTrade(self, time,open):
        #Close the trade in Train/Test Mode
        self.trade.exit_time = time
        self.trade.exit_price = open
        self.trades.append(self.trade)
        self.trade = None
