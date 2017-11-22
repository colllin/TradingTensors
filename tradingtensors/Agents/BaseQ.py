import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np

# Huber損失
# 統計学において、ロバスト回帰で使われる損失関数の一つ
#https://en.wikipedia.org/wiki/Huber_loss
def huber_loss(error, delta=1.0):
    return tf.where(
        tf.abs(error) < delta,
        tf.square(error) * 0.5,
        delta * (tf.abs(error) - 0.5 * delta)
    )

class DQN(object):
    def __init__(self, observations, actions , scope_name):
        with tf.variable_scope(scope_name):
            self.features = tf.placeholder(tf.float32, [None, observations])
            self.dropout = tf.placeholder(tf.float32)

            inputs = self.features
            for hidden in [128, 64, 32]:
                inputs = layers.fully_connected(
                    inputs=inputs,
                    num_outputs= hidden,
                    activation_fn = tf.tanh,
                    weights_regularizer = layers.l2_regularizer(scale=0.1))
                inputs = tf.nn.dropout(inputs, self.dropout)
            self.Q_t = layers.fully_connected(inputs, actions, activation_fn=None)
            self.Q_action = tf.argmax(self.Q_t, axis=1)

            # create_optimizer
            #Placeholder to hold values for Q_values estimated by target_network
            self.target_q_t = tf.placeholder(tf.float32, [None])

            #Compute current_Q estimation using online network, states and action are drawn from training batch
            self.action = tf.placeholder(tf.int64, [None])
            action_one_hot = tf.one_hot(self.action, actions, 1.0, 0.0)
            current_Q = tf.reduce_sum(self.Q_t * action_one_hot, reduction_indices=1)

            #Difference between target_network and online network estimation
            error = huber_loss(self.target_q_t - current_Q)

            self.loss = tf.reduce_mean(error)

            #Dynamic Learning steps- decaying with episodes
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(1e-3, global_step, 10000, 0.96, staircase=True)
            self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.optimize = self.trainer.minimize(self.loss, global_step=global_step)
            # create_optimizer end
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name)
class DDQN(object):
    def __init__(self, observations, actions):
        for name in ['online','target']:
            dqn = DQN(observations, actions, name)
            setattr(self, name, dqn)
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []

    def add(self, observation, action, reward, next_observation, terminal):
        instance = (observation, action, reward, next_observation, terminal)

        if len(self.storage) <= self.capacity:
            self.storage.append(instance)
            return
        #Remove the first, add to the other end
        self.storage.pop(0)
        self.storage.append(instance)

    def sample(self, size):

        #Select size indexes from storage
        index = np.random.choice(len(self.storage), size=size, replace=False)

        observation, actions, rewards, next_observation, terminal = [], [], [], [], []

        #Regroup every sample into different lists
        for i in index:
            _sample = self.storage[i]
            observation.append(_sample[0])
            actions.append(_sample[1])
            rewards.append(_sample[2])
            next_observation.append(_sample[3])
            terminal.append(_sample[4])

        observation = np.array(observation)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observation = np.array(next_observation)
        terminal = np.array(terminal)

        return observation, actions, rewards, next_observation, terminal