import csv
import numpy as np
import serial

ser = serial.Serial("COM7", 9600, timeout=None)
ser.timeout = None

line = ""
f = file("data2.csv", "a+")

while (True):
    c = ser.read()
    line = line + c.decode("utf-8")
    if c == b'\n':
        print(line)
        f.write(line)
        line = ""
