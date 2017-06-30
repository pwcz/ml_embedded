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
        if np.random.random() < 1 - self.e_greedy:
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
        return r"RL: e-greedy (e = " + str(self.e_greedy) + ")"

    def logger_info(self):
        return self.__str__()


class SARSAAlgorithm:
    def __init__(self, actions=np.array([60, 45, 35, 30, 16, 8, 4, 2, 1]), learning_rate=.5, discount_factor=.5,
                 e_greedy=.5):
        self.actions = actions
        self.states_num = 72
        self.lr = learning_rate
        self.y = discount_factor
        self.e_greedy = e_greedy
        self.q_table = np.zeros([self.states_num, self.actions.size])
        # self.q_table = np.full([self.states_num, self.actions.size], -24*60)
        # self.q_table.fill(-np.inf)
        self.current_state = None
        self.action = 0
        self.prev_action = 0

    def choose_action(self, state):
        self.prev_action = self.action
        if np.random.random() > 1 - self.e_greedy:
            self.action = np.where(self.q_table[state, :] == np.random.choice(self.q_table[state, :]))[0][0]
        else:
            self.action = np.argmax(self.q_table[state, :])
        self.current_state = state
        return self.actions[self.action]

    def update_knowledge(self, reward, next_state):
        delta = self.lr*(reward + self.y*self.q_table[next_state, self.action] -
                         self.q_table[self.current_state, self.prev_action])
        self.q_table[self.current_state, self.action] += delta

    def __str__(self):
        return r"SARSA ($\varepsilon$=" + str(self.e_greedy) + r", $\alpha$=" + str(self.lr) + r",$\gamma$=" + \
               str(self.y) + ")"

    def logger_info(self):
        return r"SARSA (e=" + str(self.e_greedy) + r",a=" + str(self.lr) + r",y=" + str(self.y) + ")"


class QLearning:
    def __init__(self, actions, learning_rate=.5, discount_factor=.5, e_greedy=.5, states_number=None):
        self.actions = actions
        self.states_num = states_number
        self.lr = learning_rate
        self.y = discount_factor
        self.e_greedy = e_greedy
        self.q_table = np.zeros([self.states_num, self.actions.size])
        self.current_state = None
        self.action = None

    def choose_action(self, state, restricted_state=None):
        if restricted_state is not None:
            q_tab = self.q_table[state, :].copy()
            search = [np.where(self.actions == x)[0][0] for x in restricted_state]
            q_tab = np.delete(q_tab, np.array(search))
            actions = self.actions.copy()
            actions = np.delete(actions, np.array(search))
            if np.random.random() > 1 - self.e_greedy:
                action = np.where(q_tab == np.random.choice(q_tab))[0][0]
            else:
                action = np.argmax(q_tab)
            self.current_state = state
            self.action = self.actions[np.where(self.actions == actions[action])]
            return actions[action]
        else:
            if np.random.random() > 1 - self.e_greedy:
                self.action = np.where(self.q_table[state, :] == np.random.choice(self.q_table[state, :]))[0][0]
            else:
                self.action = np.argmax(self.q_table[state, :])
            self.current_state = state
            return self.actions[self.action]

    def show_q_data(self):
        print("PRINT Q DATA")
        print("float Q[" + str(self.states_num) + "][5] = {")
        for state in range(self.states_num):
            actions = []
            for action in range(5):
                actions.append(str(self.q_table[state][action]))
            print("{" + ",".join(actions) + "},")
        print("}")
        # print(self.q_table)
        print("END PRINT Q DATA")
        # for x in range(self.states_num):
        #     print(self.actions[np.argmax(self.q_table[x, :])])

    def update_q_table(self, reward, next_state):
        delta = self.lr*(reward + self.y*np.max(self.q_table[next_state, :]) -
                         self.q_table[self.current_state, self.action])
        self.q_table[self.current_state, self.action] += delta

    def __str__(self):
        return r"Q-Learning ($\varepsilon$=" + str(self.e_greedy) + r", $\alpha=$" + str(self.lr) + r",$\gamma$=" + \
               str(self.y) + ")"

    def logger_info(self):
        return r"Q-Learning (e=" + str(self.e_greedy) + r",a=" + str(self.lr) + r",y=" + str(self.y) + ")"

    update_knowledge = update_q_table


class FixedDelay:
    def __init__(self, delay=3):
        self.const_delay = delay

    def choose_action(self, *_):
        return self.const_delay

    def update_q_table(self, *_):
        pass

    def __str__(self):
        return self.logger_info()

    def logger_info(self):
        return r"Stały czas wybudzania: " + str(self.const_delay) + " s"

    update_knowledge = update_q_table
