#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import serial


class ArduinoSerial:
    def __init__(self, interface='/dev/ttyACM0', speed=115200, timeout=32):
        self.connection = serial.Serial(interface, speed, timeout=timeout)
        data = self.connection.read_all()
        print(data)

    def send_action(self):
        self.connection.write(bytes(b'F'))
        data = float(self.connection.read_until().strip())
        return data

