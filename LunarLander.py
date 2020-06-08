import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random


env = gym.make("LunarLander-v2")

observations = env.reset()


# 8 observations:
#   1. horizontal coordinate
#   2. vertical coordinate
#   3. horizontal speed
#   4. vertical speed
#   5. angle
#   6. angular speed
#   7. 1 if first leg has contact, else 0
#   8. 1 if second leg has contact, else 0

# 4 actions:
#   1. Do nothing
#   2. Fire Left Engine (our right)
#   3. Fire Main Engine
#   4. Fire Right Engine (our left)

# Q'(s1, s2, ..., s8, a) = (1-w) * Q(s1, s2, ..., s8, a) + w*(r + d * max(Q(s1', s2', ..., s8', a)) - Q(s1, s2, ..., s8, a))
# w = learning rate
# d = decay rate
# r = reward
# s# = state #
# a = action

class DeepQNetwork:
    def __init__(self, env, epsilon=1.0, learn_rate=0.001, decay_rate=0.09):
        self.env = env
        self.input_shape = env.observation_space.shape
        self.output_shape = env.action_space.n
        self.epsilon = epsilon
        self.w = learn_rate
        self.d = decay_rate
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.memory = deque()
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(520, input_shape=self.input_shape, activation="relu"))
        model.add(Dense(260, activation="relu"))
        model.add(Dense(self.output_shape, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.w))
        return model

    def act(self, given_state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(given_state)[0])
        return np.random.randint(0, self.output_shape)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    # Q'(s1, s2, ..., s8, a) = (1-w) * Q(s1, s2, ..., s8, a) + w*(r + d * max(Q(s1', s2', ..., s8', a)) - Q(s1, s2, ..., s8, a))
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        dones = np.array([i[4] for i in batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.d


agent = DeepQNetwork(env)

for j in range(4000):
    state = env.reset()
    state = np.reshape(state, (1, 8))
    for i in range(3000):
        if j > 4900:
            env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, (1, 8))
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.update()

        if done:
            print(f"Timestep {j + 1}: {reward}")
            break


env.close()
