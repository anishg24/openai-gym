import gym
import numpy as np
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make("CarRacing-v0")
observations = env.reset()


class DQN:
    def __init__(self, env):
        self.env = env
        self.input_shape = env.observation_space.shape
        self.output_shape = env.action_space.shape
        self.epsilon = 1.0
        self.gamma = 0.9
        self.batch_size = 64
        self.learn_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(180, input_shape=self.input_shape, activation="relu"))
        model.add(Dense(120, activation="relu"))
        model.add(Dense(self.output_shape, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learn_rate))
        return model

    def act(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        else:
            return np.random.randint(-1, 1, size=self.output_shape)
