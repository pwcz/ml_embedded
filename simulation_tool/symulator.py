#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import datetime
import numpy as np
from copy import deepcopy


class System:
    def __init__(self, _start_time, start_power=0, test_plan=None):
        self.start_power = start_power
        self.power_left = self.start_power
        self.last_event = _start_time
        self.last_decision = _start_time
        self.time_to_delay = 600
        self.decision_interval = 1200
        self.active_input = False
        self.reward = 0
        self.time_to_read_input = 0
        self.current_clock = 100
        self.learning_a = deepcopy(test_plan.learning_module)
        self.first_start = True
        self.event_count = 0
        self.awake_count = 0
        self.ml_parameters = {'d_penalty': 150, 'd_award': 150, 's_penalty': 0, 'delay_threshold': 7.5,
                              'delay_scale': 0, 'awake_ratio_penalty': 11, 'awake_ratio_award': 1200,
                              'awake_ratio_threshold': 10}

    def simulation_step(self, _current_time):
        if self.time_to_read_input <= 0:
            self.reward -= self.ml_parameters['s_penalty']
            self.awake_count += 1
            self.power_left += 1
            self.time_to_read_input = self.current_clock
        else:
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

    def event_action(self, _current_time):
        self.event_count += 1
        self.reward -= self.ml_parameters['s_penalty']
        if self.time_to_read_input > np.random.normal(loc=self.ml_parameters['delay_threshold'],
                                                      scale=self.ml_parameters['delay_scale']):
            self.reward -= self.ml_parameters['d_penalty']
        else:
            self.reward += self.ml_parameters['d_award']

        self.reward -= self.time_to_read_input
        return self.time_to_read_input

    def calculate_reward(self):
        reward = self.reward
        #
        # #
        # try:
        #     awake_ratio = self.awake_count/self.event_count
        # except ZeroDivisionError:
        #     awake_ratio = self.awake_count
        #
        # if abs(awake_ratio - self.ml_parameters['awake_ratio_threshold']) < 10:
        #     reward += self.ml_parameters['awake_ratio_award']
        # else:
        #     reward -= self.ml_parameters['awake_ratio_penalty']
        reward += self.event_count * self.awake_count * self.ml_parameters['awake_ratio_award'] - \
                  self.awake_count * self.ml_parameters['awake_ratio_penalty']
        self.reward = 0
        self.awake_count = 0
        self.event_count = 0
        return reward


class SystemTransmission:
    def __init__(self, _start_time, start_power=0, test_plan=None):
        self.start_power = start_power
        self.power_left = self.start_power
        self.last_event = _start_time
        self.last_decision = _start_time
        self.time_to_delay = 600
        self.decision_interval = 1200
        self.active_input = False
        self.reward = 0
        self.time_to_read_input = 0
        self.current_clock = 100
        self.learning_a = deepcopy(test_plan.learning_module)
        self.first_start = True
        self.event_count = 0
        self.awake_count = 0
        self.power_consumption = {'idle': 22, 'sleep': 1.37, 'off': 0.12, 'gprs': [86.7, 403.7], 'sms': [73.8, 239.1],
                                  'reg': 684.7}
        self.timings = {'sms': 67, 'gprs': 35, 'register': 120}