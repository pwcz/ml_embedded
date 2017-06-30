#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from generate_data import DataGenerator
import datetime
from time import sleep
import numpy as np
from arduino.serial_arduino import ArduinoSerial


class ArduinoConnection:
    def __init__(self):
        self.data = DataGenerator()
        self.start_time = None
        self.arduino = ArduinoSerial()


    def test(self):
        self.set_current_time()
        print(self.start_time)
        while True:
            elapsedTime = datetime.datetime.now() - self.start_time
            print(elapsedTime.total_seconds())
            sleep(2)
            #

    def seconds_elapsed(self):
        elapsedTime = datetime.datetime.now() - self.start_time
        return elapsedTime.total_seconds()

    def test2(self):
        for x in reversed(self.data.get_train_data()):
            print(self.datetime2seconds(x))

    def set_current_time(self):
        self.start_time = datetime.datetime.now()

    def real_time(self, epoch=None):
        rt = datetime.datetime.now()
        if epoch is None:
            return str(rt.hour) + ":" + str(rt.minute) + ":" + str(rt.second) + " # "
        else:
            return "epoch: " + str(epoch) + " || " + str(rt.hour) + ":" + str(rt.minute) + ":" + str(rt.second) + " # "

    def datetime2seconds(self, date):
        if type(date) is int:
            return date
        return date.minute * 60 + date.second

    def perform_learning(self, epoch_num):
        test_set = [1, 10, 20, 35, 55]
        delays = []
        for epoch in range(epoch_num):
            self.set_current_time()
            delays_epoch = []
            # for action in test_set:
            for action in reversed(self.data.get_train_data()):
                print(self.real_time(epoch) + "Next action: " + str(action))
                while True:
                    if self.seconds_elapsed() >= self.datetime2seconds(action):
                        t_th = np.random.normal(loc=7.5, scale=2)
                        # t_th = np.random.normal(loc=3, scale=2)
                        delay = self.arduino.send_action(t_th)
                        print(self.real_time(epoch) + "Action executed (" + str(action) + "), delay : " + str(delay) +
                              ", t_th = " + str(t_th))
                        delays_epoch.append(delay)
                        with open("arduino_results/arduino_epoch_" + str(epoch) + ".txt", "a") as myfile:
                            myfile.write(str(action) + ";" + str(delay) + "\n")
                        break
                    sleep(0.1)
                    # print("wait for ", action, self.seconds_elapsed())
            with open("arduino_results/arduino_results_" + str(epoch) + ".txt", "a") as myfile:
                myfile.write(str(epoch) + ";" + str(np.mean(delays_epoch)) + "\n")
            with open("arduino_results/arduino_results.txt", "a") as myfile:
                myfile.write(str(epoch) + ";" + str(np.mean(delays_epoch)) + "\n")                
            delays.append(np.mean(delays_epoch))
            print(self.real_time(epoch) + " avg delay: " + str(np.mean(delays_epoch)))
            while True:
                if self.seconds_elapsed() >= 60*60:
                    break
        print(delays)

if __name__ == "__main__":
    dev = ArduinoConnection()
    dev.perform_learning(160)

