#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import datetime
import numpy as np


class System:
    def __init__(self, _start_time):
        self.power_consumption = {True: 0.1, False: 1}
        self.is_standby_mode = False
        self.power_left = 86400
        self.last_event = _start_time
        self.last_decision = _start_time
        self.delays = {False: 0.55, True: 3}
        self.time_to_delay = 600
        self.decision_interval = 1200
        self.active_input = False
        self.reward = 0
        self.time_to_read_input = 0
        self.current_clock = 100
        self.learning_a = QLearning()
        self.first_start = True

    def simulation_step(self, _current_time):
        if self.time_to_read_input <= 0:
            self.reward -= 1
            self.power_left -= 1
            self.time_to_read_input = self.current_clock
        else:
            self.time_to_read_input -= 1

        # self.power_left -= self.power_consumption[self.is_standby_mode]
        if _current_time - self.last_decision >= datetime.timedelta(seconds=self.decision_interval)\
                or _current_time < self.last_decision:
            # print(_current_time)
            self.last_decision = _current_time
            state = (_current_time.hour * 3600 + _current_time.minute * 60 +
                     _current_time.second) / self.decision_interval
            # print("state = " + str(state))
            if not self.first_start:
                self.learning_a.update_q_table(self.calculate_reward(), int(state))
            self.first_start = False
            self.current_clock = self.learning_a.choose_action(int(state))

    def event_action(self, _current_time):
        self.reward += 21
        self.reward -= self.time_to_read_input
        return self.time_to_read_input
        # action_delay = self.delays[self.is_standby_mode]
        # self.is_standby_mode = False
        # self.last_event = _current_time
        # return action_delay

    def calculate_reward(self):
        reward = self.reward
        self.reward = 0
        return reward


class QLearning:
    def __init__(self):
        self.actions = np.array([60, 45, 35, 30, 16, 8, 4, 2, 1])
        self.decision_time = 300
        self.states_num = 72
        self.lr = 0.5
        self.y = 0.5
        self.e_greedy = 0.05
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
