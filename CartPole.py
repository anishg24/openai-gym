import gym
import numpy as np

env = gym.make("CartPole-v0")


# Q'(s1, s2, s3, a) = (1-w) * Q(s1, s2, s3, a) + w*(r + d * max(Q(s1', s2', s3', a)) - Q(s1, s2, s3, a))


def greedy_epsilon(env, epsilon=1.0, min_eps=0, episodes=4000, learning=0.001, gamma=0.9):
    reward_target = 195
    reward_increment = 1
    reward_threshold = 0
    epsilon_delta = (epsilon - min_eps) / episodes

    position_max, _, angle_max, _ = env.observation_space.high
    position_low, _, angle_low, _ = env.observation_space.low
    state_range = np.array([position_max - position_low, angle_max - angle_low])
    descritize = lambda x: np.round(np.abs(x), 0).astype(int)
    state_range = descritize(state_range) + 1

    Q = np.random.uniform(low=-1, high=1, size=(state_range[0], state_range[1], env.action_space.n))

    for i in range(episodes):
        done = False
        score, total_score = 0, 0

        state1 = env.reset()
        state1 = np.array(state1 - env.observation_space.low)
        state1 = descritize(state1)

        while not done:
            if i > 3000:
                env.render()

            if np.random.random() > epsilon:
                action = np.argmax(Q[state1[0], state1[2]])
            else:
                action = np.random.randint(0, env.action_space.n)

            state2, reward, done, _ = env.step(action)
            state2 = np.array(state2 - env.observation_space.low)
            state2 = descritize(state2)

            delta = learning * (reward + gamma * np.max(Q[state2[0], state2[2]])) - Q[state1[0], state1[2], [action]]
            Q[state1[0], state1[2], [action]] += delta

            total_score += reward
            state1 = state2

            if epsilon > min_eps:
                epsilon -= epsilon_delta

        if (i + 1) % 100 == 0:
            print(f"Episode {i + 1}: {total_score}")


greedy_epsilon(env)
env.close()
