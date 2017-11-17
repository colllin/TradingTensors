import os
import time
from datetime import datetime
from queue import LifoQueue
from threading import Thread

import numpy as np
import pandas as pd
import tensorflow as tf

from ..settings.DQNsettings import (FINAL_P, GAMMA, INITIAL_P, UPDATE_FREQUENCY, DROPOUT)
from .BaseQ import (DQN, LinearDecay, ReplayBuffer)
from .visual_utils import ohlcPlot, rewardPlot

class DQNAgent():

    def __init__(self, env, directory):
        self.env = env

        self.model_directory =None
        self.directory = directory

        self.neurons = [128, 64, 32]
        self.online_dqn = DQN(env, self.neurons, 'online')
        self.target_dqn = DQN(env, self.neurons, 'target')

        self.best_models = []
    def update(self,session):
        #Copy variables of online network to target network
        for on_, tar_ in zip(self.online_dqn.variables, self.target_dqn.variables):
            session.run(tf.assign(tar_,on_))
    def choose_action(self, state, epsilon, session, train):
        #maintain dropout ratio if training, else keep all neurons
        if np.random.random() < epsilon:
            #Exploration
            return np.random.choice(self.env.action_space)
        #Exploitation
        dropout = DROPOUT if train else 1.0
        return session.run(
            self.online_dqn.Q_action,
            feed_dict={
                self.online_dqn.features: state[np.newaxis, :],
                self.online_dqn.dropout: dropout
                }
            )[0]
    def mini_batch_training(self, session, replaybuff, batch_size=32, discount=0.99):
        '''
        Sample Batch from memory and optimize online network
        '''
        obses_t, actions, rewards, obses_tp1, terminal = replaybuff.sample(batch_size)

        #Double Q learning implementation
        state_shape = [batch_size, self.env.observation_space]

        #Use online network to generate next actions
        next_action = session.run(self.online_dqn.Q_action, feed_dict ={
            self.online_dqn.features: np.reshape(obses_tp1, state_shape),
            self.online_dqn.dropout: DROPOUT
        })
        #Use target network to predict next Q_value
        next_Q = session.run(self.target_dqn.Q_t, feed_dict ={
                self.target_dqn.features: np.reshape(obses_tp1, state_shape),
                self.target_dqn.dropout: DROPOUT
            })

        #Select Q_values indexed by pred_actions
        Q_prime = [next_Q[i][a] for i, a in enumerate(next_action)]


        #Update Rule of the Bellman Equation
        target_q_t = rewards + (1. - terminal) * discount * Q_prime

        _ = session.run([self.online_dqn.optimize], feed_dict={
            self.online_dqn.target_q_t: target_q_t,
            self.online_dqn.action: actions,
            self.online_dqn.features: np.reshape(obses_t, state_shape),
            self.online_dqn.dropout: DROPOUT
        })
    def train(
        self,
        policy_measure='optimal',
        batch_size = 32,
        CONVERGENCE_THRESHOLD = 2000,
        EPISODES_TO_EXPLORE = 30,
        train_episodes = 200
        ):
        '''
        Run the full training cycle
        '''

        assert policy_measure in ['average', 'highest', 'optimal'], \
        "policy measure can only be 'average', 'highest', or 'optimal'"

        #Define saved model directory
        timestamp = datetime.fromtimestamp(time.time()).strftime('%H%M')
        self.model_directory = os.path.join(self.directory, timestamp+'_Episode%s.ckpt')

        #Clear all previous tensor models
        for dir_ in os.listdir(self.directory):
            os.remove(os.path.join(self.directory, dir_))

        steps_per_episode = self.env.sim.train_end_idx - self.env.sim.train_start_idx - 2

        #Create a Transition memory storage
        replaybuffer = ReplayBuffer(steps_per_episode * train_episodes * 1.2)

        #Use of parallelism
        config_proto=tf.ConfigProto(
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8
            )

        #Keep track of top 10 models
        top_10s = []

        with tf.Session(config=config_proto) as session:

            #Initialize all weights and biases in NNs
            session.run(tf.global_variables_initializer())

            #Update Target Network to Online Network
            self.update(session)

            saver = tf.train.Saver(max_to_keep=None)

            #Reseting all tools
            self.journal_record = []
            self.reward_record = []
            self.avg_reward_record = []
            self.equity_curve_record = []
            t = 0
            max_score = 0


            for episode in range(1, train_episodes+1):

                observation = self.env.reset(TRAIN=True)

                done, SOLVED = False, False

                while not done:

                    #Pick the decayed epsilon value
                    exploration = LinearDecay(
                        t,
                        EPISODES_TO_EXPLORE * steps_per_episode,
                        INITIAL_P,
                        FINAL_P)

                    #Pick an action using online network
                    action = self.choose_action(
                        observation,
                        exploration,
                        session,
                        train=True)

                    #Advance one step with the action in our environment
                    new_observation, _action, reward, done = self.env.step(action)

                    #Add the Experience to the memory
                    replaybuffer.add(observation, _action, reward, new_observation, float(done))


                    observation = new_observation
                    t += 1

                    if t > UPDATE_FREQUENCY:
                        #Optimize online network with SGD
                        self.mini_batch_training(session, replaybuffer, batch_size, GAMMA)

                    if t % UPDATE_FREQUENCY == 0:
                        #Periodically copy online net to target net
                        self.update(session)

                    if done:
                        '''End of Episode routines'''

                        #Close the Last Trade in portfolio if any
                        if self.env.portfolio.isHoldingTrade():
                            lastTime = self.env.sim.data.index[self.env.sim.curr_idx].to_pydatetime()
                            lastOpen = self.env.sim.data['Open'].iloc[self.env.sim.curr_idx]
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

                        #Save this score
                        if policy_measure == 'average':
                            score = average_pips_per_trade
                        elif policy_measure == 'highest':
                            score = self.reward_record[-1]
                        else:
                            score = np.abs(average_pips_per_trade) * self.reward_record[-1]


                        #Is this score greater than any current top 10s?

                        TERMINAL_PATH = self.model_directory % episode


                        #Condition: Only start recording this score if agent is no longer exploring
                        if episode > EPISODES_TO_EXPLORE:


                            if len(top_10s) < 10:
                                # Just append if there are not enough on the list
                                top_10s.append((episode, score))

                                #Sort the list
                                top_10s = sorted(top_10s, key=lambda x: x[1], reverse=True)

                                #Save the maximum score
                                max_score = top_10s[0][1]

                                saver.save(session, TERMINAL_PATH)
                            else:
                                replace = False
                                insertion_idx = None
                                for i, _tuple in enumerate(top_10s):
                                    _epi, _score = _tuple[0], _tuple[1]

                                    if score > _score:
                                        max_score = score
                                        insertion_idx = i
                                        replace = True
                                        break

                                #Remove from the last index, insert this
                                if replace:
                                    top_10s.pop()
                                    top_10s.insert(insertion_idx, (episode, score))
                                    saver.save(session, TERMINAL_PATH)

                        if exploration == FINAL_P and len(top_10s) == 10:

                            if np.mean(self.reward_record[-16:-1]) > CONVERGENCE_THRESHOLD:
                                SOLVED = True

                        break

                if SOLVED:
                    print ("CONVERGED!")
                    print ()
                    break

        self.best_models = top_10s


    def trainSummary(self, TOP_N=3):

        #Plot Total Reward
        rewardPlot(self.reward_record, self.best_models, 'Total', TOP_N)

        #Plot Average Reward
        rewardPlot(self.avg_reward_record, self.best_models, "Average", TOP_N)

        for i,m in enumerate(self.best_models):
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
        ohlcPlot(self.journal_record[index], self.env.sim.data, self.equity_curve_record[index])



    def test(self, episode):
        '''
        episode: int, episode to be selected
        '''
        assert len(os.listdir(self.directory)) > 0, "No saved tensor models are found for this model, please train the network"

        model_file = self.model_directory % episode

        with tf.Session() as session:
            #Create restoration path
            saver = tf.train.Saver()
            saver.restore(session, model_file)

            observation = self.env.reset(TRAIN=False)
            done = False

            while not done:

                #Select Action
                action = self.choose_action(
                        observation,
                        0, #Greedy selection
                        session,
                        train=False)

                #Transit to next state given action
                new_observation, _, _, done = self.env.step(action)

                observation = new_observation


            average_pips_per_trade = self.env.portfolio.total_reward / self.env.portfolio.total_trades
            self.journal_record.append(self.env.portfolio.journal)
            self.avg_reward_record.append(average_pips_per_trade)
            self.reward_record.append(self.env.portfolio.total_reward)
            self.equity_curve_record.append(self.env.portfolio.equity_curve)


            self.episodeReview(0)


    def liveTrading(self, episode, HISTORY=20, tradeFirst=False):
        '''
        Threaded implementation of listener and handler events
        episode: int, MODEL to be chosen
        '''
        #Set Environment and its portfolio to Live Mode
        self.env.setLive()

        #Clear up the portfolio
        self.env.portfolio.reset()

        #Initialize the time of the current incomplete candle
        #Set True if start trading on current candle, (Not Recommended during Market Close)
        if tradeFirst:
            self.env.lastRecordedTime = None
        else:
            self.env.lastRecordedTime = self.env.api_Handle.getLatestTime(self.env.SYMBOL)

        model_file = self.model_directory % episode

        #Tensorflow session with Chosen Model
        session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(session, model_file)

        #Initiate an event stack
        events_q = LifoQueue(maxsize=1)

        listenerThread = Thread(target=self.env.candleListener, args=(events_q,))
        handlerThread = Thread(target=self.newCandleHandler, args=(events_q, session))

        #Start threads
        listenerThread.start()
        handlerThread.start()


    def newCandleHandler(self, queue, session, HISTORY=20):
        '''
        Receives a new Candle event and perform action
        '''
        while True:

            if queue.empty():
                continue
            event = queue.get()

            if event != 'New Candle':
                continue

            print ("Processing New Candle")
            data, states = self.env.sim.build_data_and_states(HISTORY)

            action = self.choose_action(
                states[-1], #Latest state
                0, #Greedy selection
                session,
                train=False)

            #Initiate position with Portfolio object
            self.env.portfolio.newCandleHandler(action)

            queue.task_done()
