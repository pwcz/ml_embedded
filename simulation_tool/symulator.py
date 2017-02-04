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
        self.delays = {False: 0.55, True: 3}
        self.time_to_delay = 600

    def simulation_step(self, _current_time):
        self.power_left -= self.power_consumption[self.is_standby_mode]
        if _current_time - self.last_event > datetime.timedelta(seconds=self.time_to_delay):
            self.is_standby_mode = True

    def event_action(self, _current_time):
        action_delay = self.delays[self.is_standby_mode]
        self.is_standby_mode = False
        self.last_event = _current_time
        return action_delay


class BasicLearningAlg:
    def __init__(self):
        self.actions = np.array([1, 25, 50, 75, 100])
        self.decision_time = 300
        self.memory = np.zeros([24*60*60/self.decision_time, self.actions.size])

    def make_decision(self):
        pass

    def get_reward_info(self):
        pass
