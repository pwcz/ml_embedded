#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from math import exp
import datetime
from generate_data import DataGenerator


def quality_function(_power_consumed, _avg_delays, max_power=8600, max_delay=3):
    delays_quality = (1 - _avg_delays / max_delay) * 55
    # power_consumption_ratio = max_power / _power_consumed * 45
    power_consumption_ratio = exp((-float(_power_consumed))/float(max_power)) * 45
    return float(delays_quality) * float(power_consumption_ratio)
    # return [float(delays_quality), float(power_consumption_ratio)]


class TestSceleton:
    def __init__(self):
        pass


class TestPlan:
    def __init__(self):
        pass

    class defaultTest(TestSceleton):
        start_time = current_time = datetime.datetime(2017, 1, 2, 0, 0)
        end_time = datetime.datetime(2017, 1, 3, 0, 0)
        time_delta = datetime.timedelta(seconds=1200)
        time_delta_sym = datetime.timedelta(seconds=1)
        data_generator = DataGenerator(_delta_seconds=1200, _data_schema_type='1t')
        epoch_start = 1
        epoch_number = 200
        epoch_stop = 200
        epoch_step = 1
        start_power = 86400
        quality_fun = quality_function
        get_new_test_data_for_each_epoch = False



