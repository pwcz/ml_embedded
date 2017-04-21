#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from itertools import repeat
import numpy as np
import datetime
import time
from random import randrange
import traceback
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['errorbar.capsize'] = 3
matplotlib.rcParams['grid.linestyle'] = ':'
import matplotlib.pyplot as plt
import pandas as pd


class DataGenerator:
    def __init__(self, _delta_seconds=1200, _data_schema_type='1t', _noise_schema='1t', _with_noise=True):
        self.delta_seconds = _delta_seconds
        self.SECONDS_IN_DAY = 24*60*60
        self.resolution = self.SECONDS_IN_DAY/self.delta_seconds
        self.data_schema_type = _data_schema_type
        self.noise_schema_type = _noise_schema
        self.with_noise = _with_noise
        self.data_schema = {'1t': [(6, 1, 3), (7, 0.5, 3), (12, 3, 5), (15.5, 1, 7), (19, 1, 2), (21, 1, 3)]}
        self.noise_schema = {'1t': [(2, 3, 1), (6.5, 1.5, 2), (10, 2, 1), (14, 2, 3), (17.5, 1.5, 1), (20, 1, 2),
                                    (22.5, 1.5, 1)]}
        self.prev_data_actions = []

    def gaussian(self, _x, _mu, _sig, _amp):
        return _amp*np.exp(-np.power(_x - _mu, 2.) / (2 * np.power(_sig, 2.)))

    def square_window(self, _x, _mu, _width, _amp):
        _y = np.zeros(_x.shape[0])
        _y[np.where(abs(_x - _mu) < _width)] = _amp
        return _y

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

    def generate_data_noise(self):
        data = np.linspace(0, 24, self.resolution)
        data_noise = np.zeros(int(self.resolution), dtype=np.int)
        working_day = self.noise_schema[self.noise_schema_type]

        for mu, sig, amp in working_day:
            data_noise += self.square_window(data, mu, sig, amp).astype(int)

        return [data, data_noise]

    def generate_data_actions(self):
        # np.random.seed(int(time.time()))
        data = np.linspace(0, 24, self.resolution)
        data_actions = np.zeros(int(self.resolution), dtype=np.int)
        working_day = self.data_schema[self.data_schema_type]
        noise = self.generate_data_noise()

        for mu, sig, amp in working_day:
            data_actions += self.gaussian(data, mu, sig, amp).astype(int)

        if not self.with_noise:
            return [data, data_actions]

        while True:
            for i, n in enumerate(noise[1]):
                if n == 0:
                    continue
                r = np.random.randint(low=-n, high=n+1, dtype=np.int)
                if data_actions[i] + r >= 0:
                    data_actions[i] += r
                else:
                    data_actions[i] = 0
            if not np.array_equal(self.prev_data_actions, data_actions):
                break
        self.prev_data_actions = data_actions
        return [data, data_actions]

    def get_train_data(self):
        # np.random.seed(int(time.time()))
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
    module = DataGenerator(_with_noise=False)
    test = module.generate_data_actions()
    _noise = module.generate_data_noise()
    module.with_noise = True
    noised_data = module.generate_data_actions()
    plt.errorbar(test[0], test[1], yerr=_noise[1], linestyle='dotted', fmt='o', ecolor='g', capthick=2, marker='d',
                 markersize=1)
    plt.plot(test[0], noised_data[1], 'ro')
    plt.xlabel("czas [h]")
    plt.ylabel("Ilość użytkowników")
    plt.xlim([0, 24])
    plt.ylim([0,max(test[1]+_noise[1])+1])
    plt.xticks(np.arange(0,25,2))
    plt.grid()
    plt.savefig("training_data.png")
    plt.show()

