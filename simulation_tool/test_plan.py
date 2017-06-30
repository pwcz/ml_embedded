#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import datetime
from generate_data import DataGenerator
from learning_alg import StandardLearning, QLearning, SARSAAlgorithm, FixedDelay
import numpy as np


class TestSkeleton:
    def __init__(self, delay_award=150, awake_award=10, epoch=80, awake_penalty=20, delay_threshold=7.5,
                 states_number=72, dnn=False):
        self.start_time = self.current_time = datetime.datetime(2017, 1, 2, 0, 0)
        self.end_time = datetime.datetime(2017, 1, 2, 1, 0)
        self.time_delta = datetime.timedelta(seconds=180)
        self.time_delta_sym = datetime.timedelta(milliseconds=1000)
        self.data_generator = DataGenerator(_delta_seconds=180, _data_schema_type='1t', _noise_schema='1t',
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
        self.dnn = dnn

    def get_legend(self):
        if self.dnn:
            return str(self.learning_module) + " + SN"
        else:
            return str(self.learning_module)

    def get_info(self):
        return self.learning_module.logger_info()

    def get_type(self):
        return self.type



class TestPlan:
    def __init__(self):
        pass

    class QLearning(TestSkeleton):
        def __init__(self, e_greedy=0.5, learning_rate=.5, discount_factor=.5, delay_award=150, awake_award=10,
                     epoch=20, awake_penalty=20, delay_threshold=7.5, states_number=72, dnn=False):
            TestSkeleton.__init__(self, delay_award=delay_award, awake_award=awake_award, epoch=epoch,
                                  awake_penalty=awake_penalty, delay_threshold=delay_threshold, dnn=dnn)
            self.learning_module = QLearning(actions=np.array([12, 10, 8, 6, 4]), e_greedy=e_greedy,
                                             learning_rate=learning_rate, discount_factor=discount_factor,
                                             states_number=states_number)

    class SARSAAlgorithm(TestSkeleton):
        def __init__(self, e_greedy=0.5, learning_rate=.5, discount_factor=.5):
            TestSkeleton.__init__(self)
            self.learning_module = SARSAAlgorithm(e_greedy=e_greedy, learning_rate=learning_rate,
                                                  discount_factor=discount_factor)

    class FixedDelay(TestSkeleton):
        def __init__(self, delay=3, epoch=80):
            TestSkeleton.__init__(self, epoch=epoch)
            self.learning_module = FixedDelay(delay=delay)

