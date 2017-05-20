#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import datetime
from generate_data import DataGenerator
from learning_alg import StandardLearning, QLearning, SARSAAlgorithm, FixedDelay


class TestSkeleton:
    def __init__(self):
        self.start_time = self.current_time = datetime.datetime(2017, 1, 2, 0, 0)
        self.end_time = datetime.datetime(2017, 1, 3, 0, 0)
        self.time_delta = datetime.timedelta(seconds=1200)
        self.time_delta_sym = datetime.timedelta(milliseconds=1000)
        self.data_generator = DataGenerator(_delta_seconds=1200, _data_schema_type='1t', _noise_schema='1t',
                                            _with_noise=True)
        self.epoch_start = 1
        self.epoch_number = 80
        self.epoch_stop = 80
        self.epoch_step = 1
        self.start_power = 0
        self.get_new_test_data_for_each_epoch = True

    def get_legend(self):
        return str(self.learning_module)

    def get_info(self):
        return self.learning_module.logger_info()


class TransmissionTask:
    def __init__(self):
        self.start_time = self.current_time = datetime.datetime(2017, 1, 2, 0, 0)
        self.end_time = datetime.datetime(2017, 1, 3, 0, 0)
        self.time_delta = datetime.timedelta(seconds=1200)
        self.time_delta_sym = datetime.timedelta(milliseconds=1000)
        self.data_generator = DataGenerator(_delta_seconds=1200, _data_schema_type='1t', _noise_schema='1t',
                                            _with_noise=True)
        self.epoch_start = 1
        self.epoch_number = 200
        self.epoch_stop = 200
        self.epoch_step = 1
        self.start_power = 0
        self.get_new_test_data_for_each_epoch = True


class TestPlan:
    def __init__(self):
        pass

    class StandardTest(TestSkeleton):
        def __init__(self, e_greedy=0.5):
            TestSkeleton.__init__(self)
            self.learning_module = StandardLearning(e_greedy=e_greedy)

    class QLearning(TestSkeleton):
        def __init__(self, e_greedy=0.5, learning_rate=.5, discount_factor=.5):
            TestSkeleton.__init__(self)
            self.learning_module = QLearning(e_greedy=e_greedy, learning_rate=learning_rate,
                                             discount_factor=discount_factor)

        learning_module = QLearning()

    class SARSAAlgorithm(TestSkeleton):
        def __init__(self, e_greedy=0.5, learning_rate=.5, discount_factor=.5):
            TestSkeleton.__init__(self)
            self.learning_module = SARSAAlgorithm(e_greedy=e_greedy, learning_rate=learning_rate,
                                                  discount_factor=discount_factor)

    class FixedDelay(TestSkeleton):
        def __init__(self, delay=3):
            TestSkeleton.__init__(self)
            self.learning_module = FixedDelay(delay=delay)


class TransmissionLearning:
    def __init__(self):
        pass

    class QLearning(TransmissionTask):
        def __init__(self, e_greedy=0.5, learning_rate=.5, discount_factor=.5):
            TransmissionTask.__init__(self)
            self.learning_module = QLearning(e_greedy=e_greedy, learning_rate=learning_rate,
                                             discount_factor=discount_factor)
