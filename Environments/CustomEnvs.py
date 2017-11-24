import time

import numpy as np
import pandas as pd

from settings.serverconfig import TF_IN_SECONDS, HISTORICAL_DATA_LENGTH, TRAIN_SPLIT
from functions.planetry_functions import get_planet_coordinates
from functions.utils import OandaHandler


class OandaEnv():
    def __init__(self,
                 instrument,
                 granularity,
                 train=True,
                 other_pairs=[],
                 trade_duration=1,
                 lookback_period=0,
                 planet_data={},
                 PLANET_FORWARD_PERIOD=0):

        assert granularity in TF_IN_SECONDS.keys(), "Please use this timeframe format {}".format(TF_IN_SECONDS.keys())
        assert '_' in instrument, "Please define currency pair in this format XXX_XXX"

        self.api_Handle = OandaHandler(granularity)
        precision = self.api_Handle.get_instrument_precision(instrument)

        self.simulator = OandaSimulator(
            handle=self.api_Handle,
            instrument=instrument,
            other_pairs=other_pairs,
            lookback=lookback_period,
            isTraining=train,
            precision=precision,
            # Planet Data for Mr Peter's version
            planet_data=planet_data,
            PLANET_PERIOD=PLANET_FORWARD_PERIOD
        )

        self.portfolio = Portfolio(
            handle=self.api_Handle,
            trade_duration=trade_duration,
            instrument=instrument)

        self.isTraining = train

        self.instrument = instrument
        self.action_space = 3
        self.observation_space = self.simulator.states_dim

    def step(self, action):
        new_obs, portfolio_feed, done = self.simulator.step()
        action, reward = self.portfolio.newCandleHandler(
            action=action, TIME=portfolio_feed[0],
            OPEN=portfolio_feed[1], REWARD=portfolio_feed[2])

        return new_obs, action, reward, done

    def reset(self, train):
        self.isTraining = train
        self.simulator.isTraining = train
        observation = self.simulator.reset()
        self.portfolio.reset()

        return observation

class OandaSimulator():

    def __init__(self, **kwargs):

        self.api_Handle = kwargs['handle']

        self.instrument = kwargs['instrument']
        # Attributes to create state space
        self.other_pairs = kwargs['other_pairs']  # List of other pairs
        self.LOOKBACK = kwargs['lookback']  # how many periods to lookback
        self.planet_args = [kwargs['planet_data'], kwargs['PLANET_PERIOD']]

        # Attributes for training model
        # Percentage of data to be used for training, to be used in
        self.isTraining = kwargs['isTraining']  # Controlled by Environment

        #For Normalization
        self.train_mean = None
        self.train_std = None

        self.data, self.states = self.build_data_and_states()
        self.states_dim = self.states.shape[1]

        #To be used in every step of Simulator
        self.Open = self.data.Open.values
        self.Dates = self.data.index.to_pydatetime().tolist()

        #Reward: (CLOSE - OPEN) / (0.0001)
        precision = kwargs['precision']
        self.reward_pips = (self.data['Close'] - self.data['Open']).values / precision

        '''
        Define the first and last index of states during training
        '''
        '''Before executing, check that states is defined'''
        assert self.states is not None, "No state space!"
        data_count = self.states.shape[0]
        '''Define boundary index for training and testing'''
        self.train_start_idx = 0
        self.train_end_idx = int(TRAIN_SPLIT*data_count)
        self.test_start_idx = self.train_end_idx +1
        self.test_end_idx = data_count - 1

        self.reset() #Reset to initialize curr_idx and end_idx
    '''
    Reset Function:
    set cur_idx to the first instance
    set end_idx to the last instance

    Return the first instance
    '''
    def reset(self):
        if self.isTraining:
            self.curr_idx = self.train_start_idx
            self.end_idx = self.train_end_idx
        else:
            self.curr_idx = self.test_start_idx
            self.end_idx = self.test_end_idx
        '''Edge Case: Step function will cross boundary of data '''
        if self.curr_idx == self.end_idx:
            raise Exception("Please use more history!")

        #Return the first instance of the state space
        return self.states[self.curr_idx]
    def build_data_and_states(self):
        # Pull primary symbol from Oanda API
        primary_data = self.api_Handle.get_history(self.instrument)
        assert primary_data is not None, "primary_data is not DataFrame"

        states_df = pd.DataFrame(index=primary_data.index)

        states_df['Returns'] = primary_data['Open'].pct_change()

        #Get Return of additional pairs
        if len(self.other_pairs) > 0:

            for pair_name in self.other_pairs:
                _symbol_data = self.api_Handle.get_history(pair_name)
                assert _symbol_data is not None, "{} _symbol_data is not DataFrame".format(pair_name)
                #Attach to primary data
                states_df.loc[:, "%s_Returns"%pair_name] = _symbol_data['Open'].pct_change()

        # Shift Data if there are any lookback period
        original = states_df.copy()
        if self.LOOKBACK > 0:
            for i in range(0,self.LOOKBACK):
                _shifted = original.shift(i+1)
                states_df = states_df.join(_shifted, rsuffix="_t-{}".format(i+1))


        # Compute Planetry data (Note: get_planet_coordinates perform shifting operation)
        if not (not self.planet_args[0]): #NOT Empty dictionary
            dates = primary_data.index.to_pydatetime().tolist()
            planet_data = get_planet_coordinates(dates, self.planet_args[0], self.planet_args[1])

            states_df = states_df.join(planet_data)

        states_df.dropna(axis=0, how='any', inplace=True)

        primary_data = primary_data.loc[states_df.index.tolist(), :]

        states = self.normalize_states(states_df.values)

        return primary_data, states

    '''States Normalization'''
    def normalize_states(self, states):
        if self.train_mean is None or self.train_std is None:
            self.train_mean = np.mean(states, 0)
            self.train_std = np.std(states, 0)

        transformed = (states - self.train_mean) / self.train_std
        return transformed

    def step(self):

        reward = self.reward_pips[self.curr_idx] #Current Reward: Current Close - Close Open
        THIS_OPEN = self.Open[self.curr_idx] #Current Open
        THIS_TIME = self.Dates[self.curr_idx]

        self.curr_idx += 1
        done = self.curr_idx >= self.end_idx
        new_obs = self.states[self.curr_idx] #Next State

        return new_obs, (THIS_TIME, THIS_OPEN, reward), done



class Portfolio():
    '''
    Portfolio imposes a fixed-duration trading regime
    No StopLoss is required, trades are closed automatically once they reach the specified duration
    '''
    def __init__(self, **kwargs):

        self.trade_duration = kwargs['trade_duration']
        self.api_Handle = kwargs['handle']
        self.instrument = kwargs['instrument']
        self.reset()
    def newCandleHandler(self, action, **kwargs):
        '''
        IN Training/Testing mode, step returns action and reward
        TRAIN/TEST MODE:
        kwargs = {
            'TIME' : curr_time,
            'OPEN' :curr_open,
            'REWARD': reward
        }
        '''

        if self._isHoldingTrade:
            #Increase trade duration
            self.curr_trade['Trade Duration'] += 1

            #Check if duration limit is reached
            reached = self.curr_trade['Trade Duration'] >= self.trade_duration

            if reached:
                #Close Trade
                self.closeTrade(**kwargs)
            else:
                #Continue Holding
                return self.continueHolding(**kwargs)

        if action == 2:
            # Do Nothing
            self.equity_curve.append(self.total_reward)
            REWARD = 0
            return action, REWARD

        else:
            #TAKE A TRADE
            return self.openTrade(action=action, **kwargs)


    def openTrade(self, action, **kwargs):
        self.total_trades += 1
        #Train/Test Mode
        #Set cur_trade
        self.curr_trade['ID'] = self.total_trades
        TYPE = 'BUY' if action == 0 else 'SELL'
        self.curr_trade['Type'] = TYPE

        #Set Price and Time
        self.curr_trade['Entry Time'] = kwargs['TIME']
        self.curr_trade['Entry Price'] = kwargs['OPEN']

        #Manipulate reward
        reward = kwargs['REWARD']
        multiplier = 1.0 if self.curr_trade['Type'] == 'BUY' else -1.0
        REWARD = reward * multiplier


        #Accumulate reward
        self.curr_trade['Profit'] += REWARD
        self.total_reward += REWARD

        #Update Equity
        self.equity_curve.append(self.total_reward)

        self._isHoldingTrade = True
        return action, REWARD


    def closeTrade(self, **kwargs):
        #Close the trade in Train/Test Mode
        self.curr_trade['Exit Time'] = kwargs['TIME']
        self.curr_trade['Exit Price'] = kwargs['OPEN']

        #Recalculate reward based on this open (More accurate) thisOpen - EntryPrice
        #Or we could leave curr_trade['Profit'] = lastClose - EntryPrice
        '''
        multiplier = 1.0 if self.curr_trade['Type'] == 'BUY' else -1.0
        self.curr_trade['Profit'] = multiplier * \
        (self.curr_trade['Exit Price'] - self.curr_trade['Entry Price'])
        '''
        self.journal.append(self.curr_trade)
        self.reset_trade()

    def reset_trade(self):
        self.curr_trade = {
            'ID':0,
            'Entry Price':0,
            'Exit Price':0,
            'Entry Time':None,
            'Exit Time':None ,
            'Profit':0,
            'Trade Duration':0,
            'Type':None,
            'Symbol': self.instrument
            }
        self._isHoldingTrade = False
    def reset(self):

        #Cumulative reward in this run (in pips)
        self.total_reward = 0

        #Cumulative trades in this run
        self.total_trades = 0

        self.average_profit_per_trade = 0

        #History of cumulative reward
        self.equity_curve = [] #TO BE OUTSOURCED TO AGENT

        #Trade Profile
        self.reset_trade()
        self.journal = [] #Collection of trades

    def continueHolding(self, **kwargs):
        #Reset the action
        action = 2

        #Manipulate reward
        reward = kwargs['REWARD']
        multiplier = 1.0 if self.curr_trade['Type'] == 'BUY' else -1.0
        REWARD = reward * multiplier

        #Accumulate reward
        self.total_reward += REWARD
        self.curr_trade['Profit'] += REWARD


        #Update Equity
        self.equity_curve.append(self.total_reward)


        return action, REWARD
