import csv
import numpy as np
import serial


ser = serial.Serial("COM8", 9600, timeout=None)

mode = ""
while True:
    mode = input("mode(Up, Down, Right, Left):")
    if mode.lower() == "up" or mode.lower() == "down" or mode.lower() == "right" or mode.lower() == "left":
        break

line = mode + ","
f = open("data1.csv", "a+")
f.write("###\n")
while (True):
    c = ser.read()
    line = line + c.decode("utf-8")
    if c == b'\n':
        print(line)
        f.write(line)
        line = mode + ","
f.close()
