#%% Imports
import os
import matplotlib.pyplot as plt
import sys
import csv
import numpy as np
from scipy import signal, interpolate
import scipy.stats as stats
import copy
import math


#Load in ppgtools package.
#You may have to add the relative location of the package. Any better way?
#sys.path.insert(0, r'../../../ppgtools/') 
from ppgtools import sigpro, sigimport, sigpeaks, sigplot, save, biometrics, sigseg, sigrecover


#%% Load in files.
path = r"Example data"
loc = r"\12_channel"           
            
#Load in the data. This will return a list of Biosignal objects, one for each
#channel of data.
signals_original = sigimport.importBIN(path + loc + loc)

#Load in event markers, if they exist.
markers_original = sigimport.importEventMarkers(path + loc + loc)

#%% Initial signal processing
signals = copy.deepcopy(signals_original)
markers = copy.deepcopy(markers_original)

#Here are some examples of basic signal processing of Biosignals.

#Ex 1. Convert byte data of temperature sensor into celsius.
signals[11].fs = 25/8
signals[11].convert_TMP117()

#Ex 2. Apply a bandpass filter to the PPG channels
num_ppg_channels = 8
for i in range(0, 8):
    signals[i].filter_signal("Bandpass", [0.5, 8])

#Ex 3. Sometimes the tattoo disconnects, and no data was recorded. We can pad
#the signals to maintain proper timing of the signals.
for i in range(0, len(signals)):
    markers_pad = signals[i].pad_disconnect_events(markers, method = "hold")

#%% Plot data
plt.close('all')
sigplot.plot_biosignals(signals[0:8], markers)

#Plotting just a single channel
signals[0].plot(markers)

plt.figure()
signals[0].plot_stft()

# You can access the time series data as a Numpy array as well
ppg1_np_array = signals[0].data #Retrieving the Biosignals' time series data as a Numpy array
ppg1_t = np.linspace(0, len(ppg1_np_array) / signals[0].fs, len(ppg1_np_array))
plt.figure()
plt.plot(ppg1_t, ppg1_np_array)
plt.xlabel("t [s]")
plt.ylabel("ADC count [n]")
plt.title("Plotting with Numpy Arrays")
