import numpy as np
import pandas as pd
from settings.serverconfig import HISTORICAL_DATA_LENGTH, TRAIN_SPLIT, WHALECLUB_TOKEN_DEMO
from functions.utils import OandaHandler
# from Fx import OANDA
# from sklearn.model_selection import train_test_split
# from pywhaleclub import Client

# whaleclub = Client(WHALECLUB_TOKEN_DEMO)
# print(whaleclub.get_balance())
# print(whaleclub.list_positions('active'))
# print(whaleclub.get_markets(['BTC-USD']))


# class Observation():
#     def __init__(self,
#                  key,
#                  account_id,
#                  enviroment='practice'):
#         self.oanda = OANDA(key=key,account_id=account_id,enviroment='practice')
#     def create(self,
#                instrument,
#                granularity,
#                count,
#                lookback_period):
#         _ ,self.instrument = self.oanda.api.instruments(instruments=[instrument])
#         self.instrument = self.instrument[0]
#         params = {
#             'instrument' : instrument,
#             'granularity' : granularity,
#             "count" : count
#         };
#         _, candles = self.oanda.api.candles(params)
#         states = pd.DataFrame(index=candles.index)
#         states['open'] = candles['open'].pct_change()
#         # 平均を引いて 標準偏差で割る
#         # http://data-science.gr.jp/theory/tbs_standardization.html
#         states.open = (states.open - states.open.mean()) / states.open.std()
#
#         if lookback_period > 0:
#             original = states.copy()
#             for i in range(0,lookback_period):
#                 shifted = original.shift(i+1)
#                 states = states.join(shifted, rsuffix="_t-{}".format(i+1))
#         states.dropna(axis=0, how='any', inplace=True)
#         self.states = states
#         '''States Normalization ここでやるのと微妙に値が違う
#         # 平均を引いて 標準偏差で割る
#         # http://data-science.gr.jp/theory/tbs_standardization.html
#         self.states = (states.values - np.mean(states.values, 0)) / np.std(states.values, 0)'''
#
#         candles = candles.loc[states.index.tolist(), :]
#         candles['reward'] = ( candles.close - candles.open ) / (10 ** self.instrument.pipLocation)
#
#         self.candles = candles

def augment_ticker_history(history):
    # TODO: augment history by flipping horizontally or vertically, skewing, or otherwise
    return history

    # returns_df = pd.DataFrame(index=primary_data.index)
    #
    # returns_df['Returns'] = primary_data['Open'].pct_change()
    #
    # # Shift Data if there are any lookback period
    # if lookback_period > 0:
    #     original = returns_df.copy()
    #     for i in range(1,self.lookback_period+1):
    #         _shifted = original.shift(i)
    #         returns_df = returns_df.join(_shifted, rsuffix="_t-{}".format(i))
    #
    # returns_df.dropna(axis=0, how='any', inplace=True)
    #
    # return primary_data.loc[states_df.index.tolist(), :]

    # '''States Normalization'''
    # self.returns = (states_df.values - np.mean(states_df.values, 0)) / np.std(states_df.values, 0)

    #To be used in every step of Simulator
    # self.Open = self.data.Open.values
    # self.Dates = self.data.index.to_pydatetime().tolist()


class OandaEnv():
    def __init__(self,
                 instrument,
                 granularity,
                 trade_fee_rate,
                 training=True,
                 lookback_period=1,
                 mode='practice'):

        self.oanda_api = OandaHandler(granularity=granularity, mode=mode)
        self.training = training
        self.instrument = instrument
        self.lookback_period = lookback_period  # how many periods to lookback
        self.trade_fee_rate = trade_fee_rate

        # Pull primary symbol from Oanda API
        history = self.oanda_api.get_history(self.instrument, HISTORICAL_DATA_LENGTH)
        assert history is not None, "history is not DataFrame"

        self.volume_mean = np.mean(history.Volume)
        self.volume_std = np.std(history.Volume)
        print(self.volume_mean, self.volume_std)

        train_size = int(TRAIN_SPLIT*len(history))
        self.train_history = history[:train_size]
        self.test_history = history[train_size:]

        # self.instrument_precision = self.oanda_api.get_instrument_precision(instrument)

        #Reward: (CLOSE - open) / (0.0001)
        # self.reward_pips = (self.data['Close'] - self.data['Open']).values / self.instrument_precision

        self.reset(training=self.training)

    def num_actions(self):
        # The number of currencies, 0 to hold the first, 1 to hold the second
        return 2

    def num_features(self):
        return len(self.get_observation(0))

    def get_observation(self, session_index):
        # A session frame is considered to be in the middle of a candle which has opened but not closed.
        # So we can only report on the opening of the current frame's candle, but not the close.
        returns = self.get_returns(session_index, num_candles=self.lookback_period)
        # volumes = np.array([self.session_ticker.Volume[session_index-i-1] for i in range(self.lookback_period)])
        # volumes = (volumes - self.volume_mean) / self.volume_std
        one_hot_held_currency = [0, 0]
        one_hot_held_currency[self.held_currency] = 1
        return np.concatenate([one_hot_held_currency, [100*self.trade_fee_rate], 100*returns])#, volumes])

    def get_returns(self, session_index, num_candles=1):
        ticker = self.session_ticker
        now = session_index
        return np.array([(ticker.Open[now-lookback] - ticker.Open[now-lookback-1]) / ticker.Open[now-lookback-1] for lookback in range(num_candles)])

    def step(self, action):
        # Steps you to the next session frame.

        # Our reward will be a return percentage (1 = 100% return, -1 = -100% return).
        reward = 0

        # 1 - Perform action
        if action != self.held_currency:
            # "Pay" trade_fee_rate, update held currency
            reward -= self.trade_fee_rate
            self.trade()

        # 2 - Update session index
        self.session_index += 1

        # 3 - Calculate next state / observation
        next_observation = self.get_observation(self.session_index)

        # 4 - Calculate reward
        # The return is in terms of the first currency listed in the instrument.
        # If the ticker goes up, the first currency became more valuable compared to the second.
        return_multiplier = 1.0 if self.held_currency == 0 else -1.0
        current_return = self.get_returns(self.session_index)[0]
        reward += return_multiplier * current_return
        self.hold(reward)

        # 5 - Calculate whether this is the last step
        last_tick = len(self.session_ticker)-1
        done = self.session_index >= last_tick

        return next_observation, reward, done

        # reward = self.reward_pips[self.curr_idx] #Current Reward: Current Close - Close Open
        # THIS_OPEN = self.data['Open'][self.curr_idx] #Current Open
        # THIS_TIME = self.data.index[self.curr_idx].to_pydatetime()
        # self.curr_idx += 1
        # done = self.curr_idx >= len(self.session_ticker)
        #
        #
        # new_obs = self.states[self.curr_idx] #Next State
        #
        # action, reward = self.portfolio.newCandleHandler(
        #     action=action, time=THIS_TIME,
        #     open=THIS_OPEN, reward=reward)
        #
        # return new_obs, action, reward, done

    def reset(self, training):
        self.session_ticker = augment_ticker_history(self.train_history) if training else self.test_history

        # The `session_index` is considered to be in the middle of the specified candle.
        # So you cannot exist between candles, only within an ongoing candle.
        self.session_index = 0
        # Skip the first `lookback_period`
        self.session_index += self.lookback_period

        # TODO: Consider selecting held_currency at random
        self.held_currency = 0
        self.reset_trade()

        self.trades = []

        #Return the first instance of the state space
        return self.get_observation(self.session_index)

    def reset_trade(self):
        self.current_trade = {
            'type': 'LEFT' if self.held_currency == 0 else 'RIGHT',
            'duration': 0,
            'profit': 0
        }
        self.trades.append(self.current_trade)

    def trade(self):
        self.held_currency = 0 if self.held_currency == 1 else 1
        self.reset_trade()

    def hold(self, profit):
        self.current_trade['duration'] += 1
        self.current_trade['profit'] += profit


# class Trade():
#     def __init__(self,
#             instrument,
#             id,
#             type,
#             time,
#             open):
#         self.id = id
#         self.entry_time = time
#         self.exit_time = 0
#         self.entry_price = open
#         self.exit_price = 0
#         self.profit = 0
#         self.duration = 0
#         self.type = type
#         self.instrument = instrument

# class Portfolio():
#     '''
#     Portfolio imposes a fixed-duration trading regime
#     No StopLoss is required, trades are closed automatically once they reach the specified duration
#     '''
#     def __init__(self,trade_duration,instrument):
#
#         self.trade_duration = trade_duration
#         self.instrument = instrument
#         self.reset()
#
#     def reset(self):
#
#         #Cumulative reward in this run (in pips)
#         self.total_reward = 0
#
#         #Cumulative trades in this run
#         self.total_trades = 0
#
#         #History of cumulative reward
#         self.equity_curve = [] #TO BE OUTSOURCED TO AGENT
#
#         #Trade Profile
#         self.trade = None
#         self.trades = [] #Collection of trades
#
#     def newCandleHandler(self, action,time,open,reward):
#         if self.trade is not None:
#             #Increase trade duration
#             self.trade.duration += 1
#
#             #Check if duration limit is reached
#             if self.trade.duration >= self.trade_duration:
#                 #Close Trade
#                 self.closeTrade(time,open)
#             else:
#                 #Continue Holding
#                 reward = self._updateProperties(reward)
#                 #Reset the action 2
#                 return 2, reward
#
#         if action == 2:
#             # Do Nothing
#             self.equity_curve.append(self.total_reward)
#             return action, 0
#         #TAKE A TRADE
#         return self.openTrade(action, time,open,reward)
#
#
#     def openTrade(self, action, time,open,reward):
#         self.total_trades += 1
#         #Train/Test Mode
#         type = 'BUY' if action == 0 else 'SELL'
#         self.trade = Trade(
#             self.instrument,
#             self.total_trades,
#             type,
#             time,
#             open)
#
#         reward = self._updateProperties(reward)
#
#         return action, reward
#     def _updateProperties(self,reward):
#         #Manipulate reward
#         multiplier = 1.0 if self.trade.type == 'BUY' else -1.0
#         reward = reward * multiplier
#
#
#         #Accumulate reward
#         self.trade.profit += reward
#         self.total_reward += reward
#
#         #Update Equity
#         self.equity_curve.append(self.total_reward)
#         return reward
#
#     def closeTrade(self, time,open):
#         #Close the trade in Train/Test Mode
#         self.trade.exit_time = time
#         self.trade.exit_price = open
#         self.trades.append(self.trade)
#         self.trade = None
#
# class CurrencyShiftingPortfolio():
#     '''
#     CurrencyShiftingPortfolio tracks account balances in primary and alternative currency, and a "net worth" value in the primary currency.
#     FIXME: Consider StopLoss!!!
#     No StopLoss is required, trades are closed automatically once they reach the specified duration
#     '''
#     def __init__(self,currencies,instrument,starting_balance,starting_currency):
#
#         self.currencies = instrument.split('_')
#         self.instrument = instrument
#         self.reset()
#
#     def reset(self):
#
#         # TODO: Randomly choose starting balance and starting currency.
#         # Or does it matter?  Unit balance is fine as long as rewards are to-scale.
#         self.held_balance = 1
#         self.held_currency = self.currencies[0]
#
#         # TODO: Hold
#
#         #Cumulative reward in this run (in pips)
#         self.total_reward = 0
#
#         #Cumulative trades in this run
#         self.total_trades = 0
#
#         #History of cumulative reward
#         self.equity_curve = [] #TO BE OUTSOURCED TO AGENT
#
#         #Trade Profile
#         self.trade = None
#         self.trades = [] #Collection of trades
#
#     def newCandleHandler(self, action,time,open,reward):
#         # Close open trades if they reach the max trade duration
#         if self.trade is not None:
#             #Increase trade duration
#             self.trade.duration += 1
#
#             #Check if duration limit is reached
#             if self.trade.duration >= self.trade_duration:
#                 #Close Trade
#                 self.closeTrade(time,open)
#             else:
#                 #Continue Holding
#                 reward = self._updateProperties(reward)
#                 #Reset the action 2
#                 return 2, reward
#
#         if action == 2:
#             # Do Nothing
#             self.equity_curve.append(self.total_reward)
#             return action, 0
#         #TAKE A TRADE
#         return self.openTrade(action, time,open,reward)
#
#     def step(self, desired_currency, time, open, change):
#         if self.balance_currency != self.desired_currency
#
#
#     def openTrade(self, action, time,open,reward):
#         self.total_trades += 1
#         #Train/Test Mode
#         type = 'BUY' if action == 0 else 'SELL'
#         self.trade = Trade(
#             self.instrument,
#             self.total_trades,
#             type,
#             time,
#             open)
#
#         reward = self._updateProperties(reward)
#
#         return action, reward
#     def _updateProperties(self,reward):
#         #Manipulate reward
#         multiplier = 1.0 if self.trade.type == 'BUY' else -1.0
#         reward = reward * multiplier
#
#
#         #Accumulate reward
#         self.trade.profit += reward
#         self.total_reward += reward
#
#         #Update Equity
#         self.equity_curve.append(self.total_reward)
#         return reward
#
#     def closeTrade(self, time,open):
#         #Close the trade in Train/Test Mode
#         self.trade.exit_time = time
#         self.trade.exit_price = open
#         self.trades.append(self.trade)
#         self.trade = None
