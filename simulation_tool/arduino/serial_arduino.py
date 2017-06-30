#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import serial
from time import sleep


class ArduinoSerial:
    def __init__(self, interface='/dev/ttyACM0', speed=115200, timeout=32):
        self.connection = serial.Serial(interface, speed, timeout=timeout)
        sleep(2)
        data = self.connection.read_all()
        print("ARDUINO INIT: " + str(data))

    def send_action(self, time):
        self.connection.write(bytes(str("%.2f" % time), encoding="ascii"))
        data = float(self.connection.read_until().strip())
        # data = self.connection.read_until().strip()
        return data

