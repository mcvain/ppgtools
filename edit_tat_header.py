# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 19:05:04 2022

@author: philt
"""
import os
import csv
import sys
import numpy as np
sys.path.insert(0, r'../../ppgtools/')
from ppgtools.biosignal import BioSignal
import copy
import struct
import distutils

#Write percentage of file read to console        
last_percent_written = -10
def checkPercent(total_bytes_read, file_size):
    global last_percent_written
    
    percent_complete = total_bytes_read / file_size * 100
    
    if(int(percent_complete) % 10 == 0 and int(percent_complete) != last_percent_written):
        print(str(int(percent_complete)) + "% complete")
        last_percent_written = int(percent_complete)     
        

save = True

path = r"..\..\..\Datasets\USAARL\Session 2\Subject 3"       
loc = r"\P3 experiment start"             
save_loc = loc + " (edit)"       

loc = path + loc + loc

try:    
    file = open(loc +'.bin', "rb")        
except IOError:
    print("Could not find file \"" + loc + "\"")
    
if save:
    try:
        os.mkdir(path+save_loc)
        file_out = open(path + save_loc + save_loc + ".bin", "xb")
    except IOError:
        try:
            file_out = open(path + save_loc + save_loc + ".bin", "wb")
        except IOError:
            print(f"Could not create file {loc + '_edit.bin'}.")

    
file_size = os.path.getsize(loc +'.bin')
print("File \'" + str(loc +'.bin') + "\' size: " + str(file_size / 1000) + " kB")
    

    
total_bytes_read = 0        

i = 0
#Parse the header for signal information
byte = file.read(4)

if save:
    file_out.write(byte)

header_bytes = int.from_bytes(byte, "big")
print("\nReading file header (size: " + str(header_bytes) + " bytes) for signal information")
bytes_per_packet = 0

while i < header_bytes:
    #Index, name length, name, bytes per point, fs, bit resolution, signed/unsigned
    print()
    byte = file.read(1)
    index = int.from_bytes(byte, "big")
    print("Index: " + str(index))
    
    byte = file.read(1)
    name_len = int.from_bytes(byte, "big")
    
    byte = file.read(name_len)
    name = byte.decode("utf-8")
    print("Name: " +  name)
    
    byte = file.read(1)
    bpp = int.from_bytes(byte, "big")
    print("Bytes per point: " +  str(bpp))
    bytes_per_packet += bpp
    
    byte = file.read(4)
    fs = int.from_bytes(byte, "big")
    print("Sample rate: " +  str(fs))
    
    byte = file.read(1)
    bit_res = int.from_bytes(byte, "big")
    print("Bit resolution: " +  str(bit_res))
    
    byte = file.read(1)
    signed = bool(int.from_bytes(byte, "big"))
    print("Signed: " +  str(signed))
    
    byte = file.read(1)
    little_endian = bool(int.from_bytes(byte, "big"))
    print("Little Endian: " +  str(little_endian))
    
    if save:
        edit_sig = input(f"Edit {name}? [y/n]: ")
    
        if (edit_sig == 'y' or edit_sig == 'Y'):
            bpp = int(input("Bytes per point: "))
            fs = int(input("Sample rate: "))
            bit_res = int(input("Bit resolution: "))
            signed = distutils.util.strtobool(input("Signed (true/false):"))
            little_endian = distutils.util.strtobool(input("Little Endian (true/false): "))
            
        file_out.write(index.to_bytes(1, "big"))
        file_out.write(name_len.to_bytes(1, "big"))
        file_out.write(name.encode('utf-8'))
        file_out.write(bpp.to_bytes(1, "big"))
        file_out.write(fs.to_bytes(4, "big"))
        file_out.write(bit_res.to_bytes(1, "big"))
        file_out.write(signed.to_bytes(1  , "big"))
        file_out.write(little_endian.to_bytes(1, "big"))
    
    i+= (10 + name_len)
    
total_bytes_read += header_bytes
checkPercent(total_bytes_read, file_size)

#Get the order of which the signals are in
print("\nDetermining the package structure...")
byte = file.read(2)
order_bytes = int.from_bytes(byte, "big")
if save:
    i = 0
    file_out.write(byte)
    while i < order_bytes:
        byte = file.read(1)
        file_out.write(byte)   
        i += 1
        
total_bytes_read += order_bytes
checkPercent(total_bytes_read, file_size)

#Parse the raw data   
print("\nParsing raw data...")  

#%% Read data and save
while save == True:
    #Go through each signal
    bytes_to_read = bytes_per_packet
    byte = file.read(bytes_to_read)
    if (not byte):
        break
    file_out.write(byte)

print("Done!")

#%% Close the files. I should make this guaranteed to run later.
file.close()
file_out.close()


