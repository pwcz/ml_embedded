#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from math import exp
import datetime
from generate_data import DataGenerator


class TestSceleton:
    def __init__(self):
        pass

    start_time = current_time = datetime.datetime(2017, 1, 2, 0, 0)
    end_time = datetime.datetime(2017, 1, 3, 0, 0)
    time_delta = datetime.timedelta(seconds=1200)
    time_delta_sym = datetime.timedelta(seconds=1)
    data_generator = DataGenerator(_delta_seconds=1200, _data_schema_type='1t')
    epoch_start = 1
    epoch_number = 100
    epoch_stop = 100
    epoch_step = 1
    start_power = 24*3600
    get_new_test_data_for_each_epoch = True


class TestPlan:
    def __init__(self):
        pass

    class DefaultTest(TestSceleton):
        pass

    class QLearning(TestSceleton):
        def __init__(self, ):
            pass


