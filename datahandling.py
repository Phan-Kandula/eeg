import csv
import numpy as np
import serial

ser = serial.Serial("COM7", 9600, timeout=None)
#ser.port = input("Port(COM_): ")
#ser.baudrate = input("baudrate(9600):")
ser.timeout = None

line = ""

while (True):
    c = ser.read()
    line = line + c.decode("utf-8")
    if c == b'\n':
        print(line)
        line = ""
    #print (str(c))