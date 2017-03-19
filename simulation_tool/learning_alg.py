#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np


class StandardLearning:
    def __init__(self, actions=np.array([60, 45, 35, 30, 16, 8, 4, 2, 1]), e_greedy=0.9):
        self.actions = actions
        self.states_num = 72
        self.e_greedy = e_greedy
        self.knowledge = np.zeros([self.states_num, self.actions.size])
        self.action_counts = np.zeros([self.states_num, self.actions.size])
        self.current_state = None
        self.action = None

    def choose_action(self, state):
        if np.random.random() < self.e_greedy:
            self.action = np.argmax(self.knowledge[state, :])
        else:
            self.action = np.where(self.knowledge[state, :] == np.random.choice(self.knowledge[state, :]))[0][0]
        self.current_state = state
        self.action_counts[state, self.action] += 1
        return self.actions[self.action]

    def update_knowledge(self, reward, next_state):
        self.knowledge[self.current_state, self.action] += (1/self.action_counts[self.current_state, self.action]) * \
                                                           (reward - self.knowledge[self.current_state, self.action])

    def __str__(self):
        return "RL: e-greedy (e " \
               "= " + str(self.e_greedy) + ")"


class SARSAAlgorithm:
    def __init__(self, actions=np.array([60, 45, 35, 30, 16, 8, 4, 2, 1])):
        pass

    def choose_action(self, state):
        pass

    def update_knowledge(self, reward, next_state):
        pass


class QLearning:
    def __init__(self, actions=np.array([60, 45, 35, 30, 16, 8, 4, 2, 1]), learning_rate=.5, discount_factor=.5,
                 e_greedy=.5):
        self.actions = actions
        self.states_num = 72
        self.lr = learning_rate
        self.y = discount_factor
        self.e_greedy = e_greedy
        self.q_table = np.zeros([self.states_num, self.actions.size])
        self.current_state = None
        self.action = None

    def choose_action(self, state):
        if np.random.random() > self.e_greedy:
            self.action = np.where(self.q_table[state, :] == np.random.choice(self.q_table[state, :]))[0][0]
        else:
            self.action = np.argmax(self.q_table[state, :])
        self.current_state = state
        return self.actions[self.action]

    def update_q_table(self, reward, next_state):
        delta = self.lr*(reward + self.y*np.max(self.q_table[next_state, :]) -
                         self.q_table[self.current_state, self.action])
        self.q_table[self.current_state, self.action] += delta

    def __str__(self):
        return "Q-Learning (e=" + str(self.e_greedy) + ", a=" + str(self.lr) + ",y=" + str(self.y) + ")"

    update_knowledge = update_q_table
