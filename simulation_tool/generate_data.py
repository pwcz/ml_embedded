#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from itertools import repeat
import numpy as np
import datetime
from random import randrange
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, _delta_seconds=1200, _data_schema_type='1t'):
        self.delta_seconds = _delta_seconds
        self.SECONDS_IN_DAY = 24*60*60
        self.resolution = self.SECONDS_IN_DAY/self.delta_seconds
        self.data_schema_type = _data_schema_type
        self.data_schema = {'1t': [(6, 1, 3), (7, 0.5, 3), (12, 3, 5), (15.5, 1, 7), (19, 1, 2), (21, 1, 3)]}

    def gaussian(self, _x, _mu, _sig, _amp):
        return _amp*np.exp(-np.power(_x - _mu, 2.) / (2 * np.power(_sig, 2.)))

    def time_range(self, _start_time, _end_time, _time_delta):
        _current_time = _start_time
        time_table = []
        while _current_time < _end_time:
            time_table.append(_current_time)
            _current_time += _time_delta
        return time_table

    def rand_times(self, _start_time, _time_delta, _number):
        time_table = []
        for _ in repeat(None, _number):
            time_table.append(_start_time + datetime.timedelta(seconds=randrange(_time_delta.seconds)))
        return sorted(time_table)

    def generate_data_actions(self):
        data = np.linspace(0, 24, self.resolution)
        data_actions = np.zeros(int(self.resolution), dtype=np.int)
        working_day = self.data_schema[self.data_schema_type]

        for mu, sig, amp in working_day:
            data_actions += self.gaussian(data, mu, sig, amp).astype(int)

        return [data, data_actions]

    def get_train_data(self):
        data_actions = self.generate_data_actions()[1]
        start_time = datetime.datetime(2017, 1, 2, 0, 0)
        end_time = datetime.datetime(2017, 1, 3, 0, 0)
        time_delta = datetime.timedelta(seconds=1200)

        training_data = []
        for i, x in enumerate(self.time_range(start_time, end_time, time_delta)):
            for k in self.rand_times(x, time_delta, data_actions[i]):
                training_data.append(k)
        return training_data[::-1]


if __name__ == "__main__":
    test = DataGenerator(_delta_seconds=1200, _data_schema_type='1t').generate_data_actions()
    plt.stem(test[0], test[1])
    plt.xlabel("discrete time")
    plt.ylabel("users in time window")
    plt.savefig("training_data.png")
    plt.show()

