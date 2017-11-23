import os
import time
from datetime import datetime
from queue import LifoQueue
from threading import Thread

import numpy as np
import pandas as pd
import tensorflow as tf

from ..settings.DQNsettings import (FINAL_P, GAMMA, INITIAL_P, UPDATE_FREQUENCY, DROPOUT)
from .BaseQ import (DDQN, ReplayBuffer)
from .visual_utils import ohlcPlot, rewardPlot
class BestModels():
    def __init__(self):
        self.records = []
    def check(self,episode, score):
        #Is this score greater than any current top 10s?
        #Condition: Only start recording this score if agent is no longer exploring
        if len(self.records) < 10:
            # Just append if there are not enough on the list
            self.records.append((episode, score))

            #Sort the list
            self.records = sorted(self.records, key=lambda x: x[1], reverse=True)
            return True

        insert = None
        for i, _tuple in enumerate(self.records):
            _, _score = _tuple[0], _tuple[1]
            if score <= _score:
                continue
            insert = i
            break

        #Remove from the last index, insert this
        if insert != None:
            self.records.pop()
            self.records.insert(insert, (episode, score))
            return True
        return False
class DQNAgent():

    def __init__(self, env, directory):
        self.env = env
        self.directory = directory
        self.model_directory = os.path.join(self.directory,'episode%s.ckpt')
        self.ddqn = DDQN(env.observation_space, self.env.action_space)
        self.best_model = BestModels()
    def _episodeLoop(self,
            session,
            random_choice,
            episode,
            batch_size,
            observation,
            done,
            whole_steps,
            replaybuffer):
        if random_choice:
            action = np.random.choice(self.ddqn.actions)
        else :
            #Pick an action using online network
            action = self.ddqn.choose_action(
                observation=observation,
                session=session)

        #Advance one step with the action in our environment
        new_observation, _action, reward, done = self.env.step(action)

        #Add the Experience to the memory
        replaybuffer.add(observation, _action, reward, new_observation, float(done))

        observation = new_observation
        whole_steps += 1

        if whole_steps > UPDATE_FREQUENCY:
            #Optimize online network with SGD
            self.ddqn.mini_batch_training(session, replaybuffer, batch_size, GAMMA)

        if whole_steps % UPDATE_FREQUENCY == 0:
            #Periodically copy online net to target net
            self.ddqn.update(session)
        return done, whole_steps ,observation
    def _useRondomChoice(self,whole_steps,total_steps):
        #Pick the decayed epsilon value
        # linear decay
        '''Linearly decay epsilon'''
        if whole_steps >= total_steps:
            epsilon =  FINAL_P
        else :
            if total_steps > 0:
                difference = (FINAL_P - INITIAL_P) / total_steps
            epsilon =  INITIAL_P + difference * whole_steps

        # linear decay end
        return np.random.random_sample() < epsilon, epsilon
    def train(
        self,
        batch_size = 32,
        CONVERGENCE_THRESHOLD = 2000,
        EPISODES_TO_EXPLORE = 30,
        train_episodes = 200
        ):

        #Clear all previous tensor models
        for file in os.listdir(self.directory):
            os.remove(os.path.join(self.directory, file))

        steps_per_episode = self.env.simulator.train_end_idx - self.env.simulator.train_start_idx - 2
        total_steps = EPISODES_TO_EXPLORE * steps_per_episode
        #Create a Transition memory storage
        replaybuffer = ReplayBuffer(steps_per_episode * train_episodes * 1.2)

        #Use of parallelism
        config_proto=tf.ConfigProto(
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
            )

        session = tf.Session(config=config_proto)

        #Initialize all weights and biases in NNs
        session.run(tf.global_variables_initializer())

        #Update Target Network to Online Network
        self.ddqn.update(session)

        #Reseting all tools
        self.journal_record = []
        self.reward_record = []
        self.avg_reward_record = []
        self.equity_curve_record = []
        whole_steps = 0

        for episode in range(1, train_episodes+1):
            observation = self.env.reset(train=True)
            done, solved = False, False
            while not done:
                random_choice, exploration = self._useRondomChoice(whole_steps,total_steps)
                done, whole_steps, observation = self._episodeLoop(
                    session,
                    random_choice,
                    episode,
                    batch_size,
                    observation,
                    done,
                    whole_steps,
                    replaybuffer)
                if not done :
                    continue

                #Close the Last Trade in portfolio if any
                if self.env.portfolio._isHoldingTrade:
                    lastTime = self.env.simulator.data.index[self.env.simulator.curr_idx].to_pydatetime()
                    lastOpen = self.env.simulator.data['Open'].iloc[self.env.simulator.curr_idx]
                    self.env.portfolio.closeTrade(TIME=lastTime, OPEN=lastOpen)

                #Update Bookkeeping Tools
                average_pips_per_trade = self.env.portfolio.total_reward / self.env.portfolio.total_trades
                self.journal_record.append(self.env.portfolio.journal)
                self.avg_reward_record.append(average_pips_per_trade)
                self.reward_record.append(self.env.portfolio.total_reward)
                self.equity_curve_record.append(self.env.portfolio.equity_curve)


                #Print statements at the end of every statements
                print("End of Episode %s, Total Reward is %s, Average Reward is %.3f"%(
                    episode,
                    self.env.portfolio.total_reward,
                    average_pips_per_trade
                    ))
                print("Percentage of time spent on exploring (Random Action): %s %%"%(
                    int(100 * exploration)))

                #Is this score greater than any current top 10s?
                #Condition: Only start recording this score if agent is no longer exploring
                if episode > EPISODES_TO_EXPLORE and \
                    self.best_model.check(episode, average_pips_per_trade):
                    saver = tf.train.Saver(max_to_keep=None)
                    saver.save(session, self.model_directory % episode)
                if exploration == FINAL_P and \
                    len(self.best_model.records) == 10 and \
                    np.mean(self.reward_record[-16:-1]) > CONVERGENCE_THRESHOLD:
                    solved = True
            if solved:
                print ("CONVERGED!")
                break
    def trainSummary(self, TOP_N=3):

        #Plot Total Reward
        rewardPlot(self.reward_record, self.best_model.records, 'Total', TOP_N)

        #Plot Average Reward
        rewardPlot(self.avg_reward_record, self.best_model.records, "Average", TOP_N)

        for i,m in enumerate(self.best_model.records):
            episode = m[0]
            print ("########   RANK {}   ###########".format(i+1))
            print ("Episode          | {}".format(episode))
            print ("Total Reward     | {0:.2f}".format(self.reward_record[episode-1]))
            print ("Average Reward   | {0:.2f}".format(self.avg_reward_record[episode-1]))

    def episodeReview(self, episode):

        index = episode - 1

        journal = pd.DataFrame(self.journal_record[index])

        buys = journal.loc[journal['Type']=='BUY', :]
        sells = journal.loc[journal['Type']=='SELL', :]

        print ("Summary Statistics for Episode %s \n"%(episode))
        print ("Total Trades            | {}        (Buy){}       (Sell){} "\
            .format(journal.shape[0], buys.shape[0], sells.shape[0]))

        #Calculate Profit breakdown
        total_profit = journal.Profit.sum()
        buy_profit = buys.Profit.sum()
        sell_profit = sells.Profit.sum()

        print ("Profit (in pips)        | %.2f   (Buy)%.2f   (Sell)%.2f"\
            %(total_profit, buy_profit, sell_profit))

        #Calculate Win Ratio
        total_percent = (journal.loc[journal['Profit']>0,'Profit'].count()/ journal.shape[0]) * 100
        buy_percent = (buys.loc[buys['Profit']>0, 'Profit'].count()/buys.shape[0]) * 100
        sell_percent = (sells.loc[sells['Profit']>0, 'Profit'].count()/sells.shape[0]) * 100
        print ("Win Ratio               | %.2f%%    (Buy)%.2f%%   (Sell)%.2f %%"%(total_percent, buy_percent, sell_percent))

        duration = journal['Trade Duration'].mean()
        print ("Average Trade Duration  | %.2f"%(duration))

        #print candle_stick
        ohlcPlot(self.journal_record[index], self.env.simulator.data, self.equity_curve_record[index])

    def test(self, episode):
        '''
        episode: int, episode to be selected
        '''
        assert len(os.listdir(self.directory)) > 0, "No saved tensor models are found for this model, please train the network"

        session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(session, self.model_directory % episode)

        observation = self.env.reset(train=False)
        while True:
            #Select Action
            action = self.ddqn.choose_action(
                    observation=observation,
                    session=session,
                    test=True)

            #Transit to next state given action
            observation, _, _, done = self.env.step(action)
            if done:
                break;

        average_pips_per_trade = self.env.portfolio.total_reward / self.env.portfolio.total_trades
        self.journal_record.append(self.env.portfolio.journal)
        self.avg_reward_record.append(average_pips_per_trade)
        self.reward_record.append(self.env.portfolio.total_reward)
        self.equity_curve_record.append(self.env.portfolio.equity_curve)
        self.episodeReview(0)
