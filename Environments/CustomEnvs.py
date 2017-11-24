import numpy as np
import pandas as pd

from settings.serverconfig import HISTORICAL_DATA_LENGTH, TRAIN_SPLIT
from functions.planetry_functions import get_planet_coordinates
from functions.utils import OandaHandler


class OandaEnv():
    def __init__(self,
                 instrument,
                 granularity,
                 training=True,
                 other_pairs=[],
                 trade_duration=1,
                 lookback_period=0):

        self.simulator = OandaSimulator(
            granularity=granularity,
            instrument=instrument,
            other_pairs=other_pairs,
            lookback=lookback_period,
            training=training,
        )
        self.portfolio = Portfolio(
            trade_duration=trade_duration,
            instrument=instrument)

        self.training = training
        self.observation_space = self.simulator.states_dim

    def step(self, action):
        observation, portfolio_feed, done = self.simulator.step()

        action, reward = self.portfolio.newCandleHandler(
            action=action, time=portfolio_feed[0],
            open=portfolio_feed[1], reward=portfolio_feed[2])

        return observation, action, reward, done

    def reset(self, training):
        self.training = training
        self.portfolio.reset()
        self.simulator.training = training
        return self.simulator.reset()

class OandaSimulator():

    def __init__(self, granularity, instrument, other_pairs, lookback, training):
        self.api_Handle = OandaHandler(granularity)

        self.instrument = instrument
        # Attributes to create state space
        self.other_pairs = other_pairs  # List of other pairs
        self.lookback = lookback  # how many periods to lookback

        # Attributes for training model
        # Percentage of data to be used for training, to be used in
        self.training = training  # Controlled by Environment

        #For Normalization
        self.train_mean = None
        self.train_std = None

        # Pull primary symbol from Oanda API
        primary_data = self.api_Handle.get_history(self.instrument,HISTORICAL_DATA_LENGTH)
        assert primary_data is not None, "primary_data is not DataFrame"

        states_df = pd.DataFrame(index=primary_data.index)

        states_df['Returns'] = primary_data['Open'].pct_change()

        #Get Return of additional pairs
        if len(self.other_pairs) > 0:
            for pair_name in self.other_pairs:
                _symbol_data = self.api_Handle.get_history(pair_name,HISTORICAL_DATA_LENGTH)
                assert _symbol_data is not None, "{} _symbol_data is not DataFrame".format(pair_name)
                #Attach to primary data
                states_df.loc[:, "%s_Returns"%pair_name] = _symbol_data['Open'].pct_change()

        # Shift Data if there are any lookback period
        original = states_df.copy()
        if self.lookback > 0:
            for i in range(0,self.lookback):
                _shifted = original.shift(i+1)
                states_df = states_df.join(_shifted, rsuffix="_t-{}".format(i+1))

        states_df.dropna(axis=0, how='any', inplace=True)

        self.data = primary_data.loc[states_df.index.tolist(), :]

        '''States Normalization'''
        self.states = (states_df.values - np.mean(states_df.values, 0)) / np.std(states_df.values, 0)

        self.states_dim = self.states.shape[1]

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

        self.reset() #Reset to initialize curr_idx and end_idx
    def reset(self):
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


    def step(self):

        reward = self.reward_pips[self.curr_idx] #Current Reward: Current Close - Close Open
        THIS_OPEN = self.Open[self.curr_idx] #Current Open
        THIS_TIME = self.Dates[self.curr_idx]
        self.curr_idx += 1
        done = self.curr_idx >= self.end_idx
        new_obs = self.states[self.curr_idx] #Next State

        return new_obs, (THIS_TIME, THIS_OPEN, reward), done

class Trade():
    def __init__(self, instrument):
        self.id = 0
        self.entry_time = 0
        self.exit_time = 0
        self.entry_price = 0
        self.exit_price = 0
        self.profit = 0
        self.duration = 0
        self.type = None
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
        self.trade = Trade(self.instrument)

        self.trade.id = self.total_trades
        self.trade.type = 'BUY' if action == 0 else 'SELL'

        #Set Price and Time
        self.trade.entry_time = time
        self.trade.entry_price = open

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
