#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import datetime
import numpy as np
from copy import deepcopy
import sys
import logging


class System:
    def __init__(self, _start_time, start_power=0, test_plan=None):
        self.start_power = start_power
        self.power_left = self.start_power
        self.last_event = _start_time
        self.last_decision = _start_time
        self.decision_interval = 1200
        self.reward = 0
        self.time_to_read_input = 0
        self.learning_a = deepcopy(test_plan.learning_module)
        self.current_clock = self.learning_a.choose_action(0)
        self.first_start = True
        self.event_count = 0
        self.awake_count = 0
        self.ml_parameters = {'d_penalty': test_plan.delay_award, 'd_award': test_plan.delay_award,
                              's_penalty': test_plan.awake_penalty, 's_award': test_plan.awake_award,
                              'delay_threshold': test_plan.delay_threshold, 'delay_scale': 2}
        self.power_consumption = {'sleep': 4.42, 'active': 63.68}
        self.button_push_counter = 0
        self.voltage = 6.54
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


class SystemTransmission:
    def __init__(self, _start_time, start_power=0, test_plan=None):
        self.start_power = start_power
        self.power_left = self.start_power
        self.last_event = _start_time
        self.last_decision = _start_time
        self.decision_interval = 1200
        self.reward = 0
        self.power_reward = 0
        self.time_to_read_input = 0
        self.learning_a = deepcopy(test_plan.learning_module)
        self.update_knowledge = False
        self.event_count = 0
        self.awake_count = 0
        self.power_consumption = {'idle': 63.68, 'sleep': 4.42, 'off': None,
                                  'gprs': [32.75, 36.39, 37.26, 38.82, 41.09, 42.99, 44.9, 46.81, 48.71, 50.62],
                                  'sms': 16.37}
        self.timings = {'sms': 18.16, 'gprs': [23.41, 25.13, 26.06, 27.19, 28.52, 29.74, 30.97, 32.3, 33.42, 34.65],
                        'deregister_sms': 2., 'deregister_gprs': 2.02}
        self.rewards = {'delay_award': 150, 'delay_penalty': 150, 'buff_overflow': 10000, 'lost_message': 1000}
        self.rewards_const = {'message_threshold': 500, 'power_scale': .2}
        self.current_state = 'off'
        self.max_buff_size = 9
        self.buff = []
        self.state_number = int(((24 * 60 * 60) / self.decision_interval) * (self.max_buff_size + 1))
        self.delays = [[] for _ in range(self.state_number)]
        self.button_push_counter = 0
        self.action_time = None
        self.last_action = None
        self.aggregation_count = 0

    def simulation_step(self, _current_time):
        for mgs in self.buff:
            delay = self.datatime2seconds(_current_time) - self.datatime2seconds(mgs)
            if delay < 0:
                delay = 86400 - self.datatime2seconds(mgs) + self.datatime2seconds(_current_time)
                logging.error("rtime = " + str(_current_time) + "mgs = " + str(mgs))
            if delay > 360:
                print(self.buff)
                logging.error("TIMEOUT! rtime = " + str(_current_time) + "mgs = " + str(mgs))
                self.send_gprs(_current_time, add2buff=False)
                self.learning_a.update_knowledge(-200, self.time2state(_current_time))
                self.update_knowledge = False



    def reset_epoch(self):
        self.delays = [[] for _ in range(self.state_number)]
        self.power_left = self.start_power
        self.button_push_counter = 0

    def event_action(self, _current_time):
        state = self.time2state(_current_time)
        if self.update_knowledge:
            self.learning_a.update_knowledge(self.calculate_reward(_current_time), int(state))
        self.update_knowledge = True
        if len(self.buff) == 0:
            action = self.learning_a.choose_action(state, restricted_state=[2])
        if len(self.buff) > 0:
            action = self.learning_a.choose_action(state, restricted_state=[0])
        logging.info("ACTION == " + str(action))
        self.execute_action(action, _current_time)

    def execute_action(self, action, time):
        """
        # 0 - wyślij SMS
        # 1 - dodaj wiadomość do bufora
        # 2 - wyślij przez GPRS wiadomości z bufora
        :param action:
        :param time:
        """
        if action == 0:
            self.send_sms(time)
        elif action == 1:
            if len(self.buff) > self.max_buff_size - 2:
                # logging.info("BUFFER overflow, buff size=" + str(len(self.buff)))
                self.send_gprs(time)
            else:
                self.buffer_data(time)
        elif action == 2:
            self.send_gprs(time)

    def buffer_data(self, r_time):
        self.last_action = 'buff'
        if len(self.buff) == 0:
            self.action_time = r_time
            self.buff.append(r_time)
        else:
            self.buff.append(r_time)

    def send_sms(self, r_time):
        self.last_action = 'sms'
        self.action_time = r_time
        time = self.timings['sms'] - self.timings['deregister_sms']
        energy = self.power_consumption['sms']  # power consumed for sending sms
        self.power_left += energy
        if time < 0:
            sys.exit()
        self.delays[self.time2state(r_time)].append(time)

    def send_gprs(self, r_time, add2buff=True):
        self.aggregation_count = len(self.buff)
        self.last_action = 'gprs'
        if add2buff:
            self.buff.append(r_time)
        messages_number = len(self.buff)
        time = self.timings['gprs'][messages_number] - self.timings['deregister_gprs']
        energy = self.power_consumption['gprs'][messages_number]
        self.power_left += energy
        for mgs in self.buff:
            delay = self.datatime2seconds(r_time) - self.datatime2seconds(mgs) + time
            if delay > 14400:
                logging.error("rtime = " + str(r_time) + "mgs = " + str(mgs))
            if delay < 0:
                delay = 86400 - self.datatime2seconds(mgs) + self.datatime2seconds(r_time) + time
            self.delays[self.time2state(r_time)].append(delay)
            # self.reward += self.delay2reward(delay)
        self.buff.clear()

    def calculate_reward(self, r_time):
        if self.dates2seconds(r_time, self.action_time) > 350:
            reward = 100
        else:
            if self.last_action == 'buff':
                reward = 100
            else:
                reward = -100

        if self.last_action == 'gprs':
            if self.aggregation_count < 4:
                reward -= 200
            else:
                reward += 200
        # self.action_time = r_time
        return reward

    def delay2reward(self, seconds):
        if seconds > self.rewards_const['message_threshold']:
            self.button_push_counter += 1
            return self.rewards['delay_penalty']
        else:
            return self.rewards['delay_award']

    def time2state(self, _current_time):
        time_state = int((_current_time.hour * 3600 + _current_time.minute * 60 + _current_time.second) /
                         self.decision_interval)
        val = time_state + 72 * len(self.buff)
        return val

    def datatime2seconds(self, _current_time):
        return _current_time.hour * 3600 + _current_time.minute * 60 + _current_time.second

    def dates2seconds(self, first_time, second_time):
        res_time = self.datatime2seconds(first_time) - self.datatime2seconds(second_time)
        if res_time < 0:
            res_time = 86400 - self.datatime2seconds(second_time) + self.datatime2seconds(first_time)
        return res_time

