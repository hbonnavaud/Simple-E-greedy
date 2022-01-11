import math
from statistics import mean

import gym
import gym.envs.toy_text.frozen_lake
import random
import matplotlib
import matplotlib.pyplot as plt
plt.close(plt.figure(1))
plt.ion()
fig, ax = plt.subplots()

env = gym.make('FrozenLake-v0')
q_tab = {}
episode_id = 0
gamma = 0.99
epsilon = 0.04
alpha = 0.03
success_memory = []
epsilon_memory = []
last_20_success_memory = []
while True:
    done = False
    rewards = []
    epsilon = min((1, 1 / ((episode_id + 1)/10)))
    epsilon_memory.append(epsilon)
    state = env.reset()
    print("state = ", state)
    while not done:
        # Choose action
        if not str(state) in q_tab.keys() or random.random() < epsilon:
            # Explore
            action = env.action_space.sample()
        else:
            actions_values = [value for _, value in q_tab[str(state)].items()]
            if sum(actions_values) == 0:
                action = env.action_space.sample()
            else:
                max_action_value = None
                action = None
                for a, value in q_tab[str(state)].items():
                    if max_action_value is None or max_action_value < value:
                        max_action_value = value
                        action = a
                action = int(action)
        # Step
        next_state, reward, done, _ = env.step(action)
        if reward == 0 and done:
            reward = -1
        # print("state =", state, ", action =", action, ", next_state =", next_state, ", reward =",
        # reward, ", done =", done)
        if not str(state) in q_tab.keys():
            q_tab[str(state)] = {}
        if not str(action) in q_tab[str(state)].keys():
            q_tab[str(state)][str(action)] = 0.0
        if not str(next_state) in q_tab.keys():
            max_next_action_value = 0.0
        else:
            max_next_action_value = None
            for _, value in q_tab[str(next_state)].items():
                if max_next_action_value is None or max_next_action_value < value:
                    max_next_action_value = value
        q_tab[str(state)][str(action)] = (1 - alpha) * q_tab[str(state)][str(action)] + \
            alpha * (reward + gamma * max_next_action_value)
        state = next_state
        rewards.append(reward)
        if done:
            # for state, value in q_tab.items():
            #     print("state: ", state, ", actions: ", value)
            # print("Episode " + str(episode_id) + ", rewards = " + str(sum(rewards)))
            # env.render()
            success_memory.append(sum(rewards))
            if len(success_memory) >= 20:
                last_20_success_memory.append(mean(success_memory[-20:]))
            episode_id += 1

            ax.clear()
            ax.plot(last_20_success_memory)
            ax.plot(epsilon_memory, c="red", label="e=" + str(epsilon_memory[-1]))
            ax.legend()
            plt.pause(0.00001)
