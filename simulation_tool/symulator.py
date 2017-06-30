#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import datetime
import numpy as np
from copy import deepcopy
import sys
import logging
import math

class System:
    def __init__(self, _start_time, start_power=0, test_plan=None):
        self.start_power = start_power
        self.power_left = self.start_power
        self.last_event = _start_time
        self.last_decision = _start_time
        self.decision_interval = 90  # (24*3600) / test_plan.state_number
        self.reward = 0
        self.useDNN = test_plan.dnn
        self.time_to_read_input = 0
        self.learning_a = deepcopy(test_plan.learning_module)
        self.current_clock = self.learning_a.choose_action(0)
        self.first_start = True
        self.event_count = 0
        self.awake_count = 0
        self.ml_parameters = {'d_penalty': test_plan.delay_award, 'd_award': test_plan.delay_award,
                              's_penalty': test_plan.awake_penalty, 's_award': test_plan.awake_award,
                              'delay_threshold': test_plan.delay_threshold, 'delay_scale': 2}
        # self.power_consumption = {'sleep': 4.42, 'active': 63.68}
        self.power_consumption = {'sleep': 1.5809, 'active': 33.298}
        self.button_push_counter = 0
        self.voltage = 6.0363
        self.b_event_action = False
        print("d_penalty = ", self.ml_parameters['d_penalty'], "s_penalty", self.ml_parameters['s_penalty'],
              's_award', self.ml_parameters['s_award'], "delay_threshold", self.ml_parameters['delay_threshold'])

    def simulation_step(self, _current_time):
        if self.time_to_read_input <= 0:
            if self.b_event_action:
                self.reward += self.ml_parameters['s_award']
                self.b_event_action = False
            else:
                self.reward -= self.ml_parameters['s_penalty']
            self.awake_count += 1
            self.power_left += self.power_consumption['active'] * self.voltage / 1000.
            self.time_to_read_input = self.current_clock
        else:
            self.power_left += self.power_consumption['sleep'] * self.voltage / 1000.
            self.time_to_read_input -= 1

        if _current_time - self.last_decision >= datetime.timedelta(seconds=self.decision_interval)\
                or _current_time < self.last_decision:

            self.last_decision = _current_time
            state = (_current_time.hour * 3600 + _current_time.minute * 60 +
                     _current_time.second) / self.decision_interval
            if not self.first_start:
                # update (Q table)/knowledge
                self.learning_a.update_knowledge(self.calculate_reward(), int(state))
            self.first_start = False
            # Learning - choose action
            self.current_clock = self.learning_a.choose_action(int(state))

    def reset_epoch(self):
        self.power_left = self.start_power
        self.button_push_counter = 0
        self.event_count = 0
        self.awake_count = 0
        # try:
        #     self.learning_a.e_greedy *= .9
        # except AttributeError:
        #     pass

    def event_action(self, _):
        self.event_count += 1
        self.b_event_action = True
        if self.time_to_read_input > np.random.normal(loc=self.ml_parameters['delay_threshold'],
                                                      scale=self.ml_parameters['delay_scale']):
            self.reward -= self.ml_parameters['d_penalty']
            self.button_push_counter += 1
        else:
            self.reward += self.ml_parameters['d_award']

        self.reward -= self.time_to_read_input
        return self.time_to_read_input

    def calculate_reward(self):
        reward = self.reward
        self.reward = 0
        self.awake_count = 0
        self.event_count = 0
        return reward


