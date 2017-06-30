#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import datetime
from generate_data import DataGenerator
from learning_alg import StandardLearning, QLearning, SARSAAlgorithm, FixedDelay
import numpy as np


class TestSkeleton:
    def __init__(self, delay_award=150, awake_award=10, epoch=80, awake_penalty=20, delay_threshold=7.5,
                 states_number=72):
        self.start_time = self.current_time = datetime.datetime(2017, 1, 2, 0, 0)
        self.end_time = datetime.datetime(2017, 1, 3, 0, 0)
        self.time_delta = datetime.timedelta(seconds=1200)
        self.time_delta_sym = datetime.timedelta(milliseconds=1000)
        self.data_generator = DataGenerator(_delta_seconds=1200, _data_schema_type='1t', _noise_schema='1t',
                                            _with_noise=True)
        self.epoch_start = 1
        self.epoch_number = epoch
        self.epoch_stop = self.epoch_number
        self.epoch_step = 1
        self.start_power = 0
        self.get_new_test_data_for_each_epoch = True
        self.type = "power_management"
        self.delay_award = delay_award
        self.awake_award = awake_award
        self.awake_penalty = awake_penalty
        self.delay_threshold = delay_threshold
        self.state_number = states_number

    def get_legend(self):
        return str(self.learning_module)

    def get_info(self):
        return self.learning_module.logger_info()

    def get_type(self):
        return self.type


class TransmissionTask:
    def __init__(self, epoch_number=150, lmode='ml'):
        self.start_time = self.current_time = datetime.datetime(2017, 1, 2, 0, 0)
        self.end_time = datetime.datetime(2017, 1, 3, 0, 0)
        self.time_delta = datetime.timedelta(seconds=1200)
        self.time_delta_sym = datetime.timedelta(milliseconds=1000)
        self.data_generator = DataGenerator(_delta_seconds=1200, _data_schema_type='1t', _noise_schema='1t',
                                            _with_noise=True)
        self.epoch_start = 1
        self.epoch_number = epoch_number
        self.epoch_stop = epoch_number
        self.epoch_step = 1
        self.start_power = 0
        self.get_new_test_data_for_each_epoch = True
        self.type = "transmission"
        self.mode = lmode

    def get_legend(self):
        if self.mode == 'sms':
            return r"Tylko wiadomości SMS"
        elif self.mode == 'timeout':
            return r"Staly czas agregacji: 20 minut"
        else:
            return str(self.learning_module)

    def get_info(self):
        return self.learning_module.logger_info()

    def get_type(self):
        return self.type


class TestPlan:
    def __init__(self):
        pass

    class StandardTest(TestSkeleton):
        def __init__(self, e_greedy=0.5):
            TestSkeleton.__init__(self)
            self.learning_module = StandardLearning(e_greedy=e_greedy)

    class QLearning(TestSkeleton):
        def __init__(self, e_greedy=0.5, learning_rate=.5, discount_factor=.5, delay_award=150, awake_award=10,
                     epoch=20, awake_penalty=20, delay_threshold=7.5, states_number=72):
            TestSkeleton.__init__(self, delay_award=delay_award, awake_award=awake_award, epoch=epoch,
                                  awake_penalty=awake_penalty, delay_threshold=delay_threshold)
            self.learning_module = QLearning(actions=np.array([12, 10, 8, 6, 4]), e_greedy=e_greedy,
                                             learning_rate=learning_rate, discount_factor=discount_factor,
                                             states_number=states_number)

        # learning_module = QLearning()

    class SARSAAlgorithm(TestSkeleton):
        def __init__(self, e_greedy=0.5, learning_rate=.5, discount_factor=.5):
            TestSkeleton.__init__(self)
            self.learning_module = SARSAAlgorithm(e_greedy=e_greedy, learning_rate=learning_rate,
                                                  discount_factor=discount_factor)

    class FixedDelay(TestSkeleton):
        def __init__(self, delay=3, epoch=80):
            TestSkeleton.__init__(self, epoch=epoch)
            self.learning_module = FixedDelay(delay=delay)


class TransmissionLearning:
    def __init__(self):
        pass

    class QLearning(TransmissionTask):
        def __init__(self, actions=np.array([0, 1, 2]), e_greedy=0.5, learning_rate=.5, discount_factor=.5, lmode='ml',
                     epoch_number=150):
            TransmissionTask.__init__(self, lmode=lmode, epoch_number=epoch_number)
            self.learning_module = QLearning(actions=actions, e_greedy=e_greedy, learning_rate=learning_rate,
                                             discount_factor=discount_factor, states_number=1440)
