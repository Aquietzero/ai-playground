import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import numpy as np
from itertools import cycle, count

import gc
import time

class NFQ():
    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs

    def init_model(num_inputs, num_actions, num_hidden=128):
        inputs = keras.layers.Input(shape=(num_inputs,))
        common = keras.layers.Dense(num_hidden, activation="relu")(inputs)
        action = keras.layers.Dense(num_actions, activation="softmax")(common)
        critic = keras.layers.Dense(1)(common)

        model = keras.Model(inputs=inputs, outputs=[action, critic])
        model.summary()

    def optimize_model(self, experiences, tape):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        q_sp = self.model(next_states)
        max_a_q_sp = q_sp.max(1)[0]
        target_q_s = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.model(states)

        td_errors = q_sa - target_q_s
        value_loss = td_errors.pow(2).mul(0.5).mean()
        grads = tape.gradient(value_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.model, state)
        new_state, reward, done, truncated, info = env.step(action)
        is_failure = done and not truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.experiences.append(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)

        return new_state, done

    def train(self, seed, gamma, max_episodes):
        self.gamma = gamma

        env = gym.make("CartPole-v1")

        # init model and optimizer
        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.model = self.init_model(nS, nA)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.01)

        # init history
        self.episode_reward = []
        self.episode_seconds = []
        self.episode_timestep = []
        self.episode_exploration = []
        self.evaluation_scores = []

        self.experiences = []

        with tf.GradientTape() as tape:
            for episode in range(1, max_episodes + 1):
                episode_start = time.time()

                state, _ = env.reset(seed=seed)
                is_terminal = False
                self.episode_reward.append(0.0)
                self.episode_timestep.append(0.0)
                self.episode_exploration.append(0.0)

                for step in count():
                    state, done = self.interaction_step(state, env)

                    if len(self.experiences) >= self.batch_size:
                        experiences = np.array(self.experiences)
                        batches = [np.vstack(sars) for sars in experiences.T]
                        # experiences = self.model.load(batches)
                        for _ in range(self.epochs):
                            self.optimize_model(batches, tape)

                    if done:
                        gc.collect()
                        break