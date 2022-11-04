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
import pandas as pd 


#Load in ppgtools package.
#You may have to add the relative location of the package. Any better way?
#sys.path.insert(0, r'../../../ppgtools/') 
from ppgtools import sigpro, sigimport, sigpeaks, sigplot, save, biometrics, sigseg, sigrecover


#%% Load in files.
path = r"data/eeg"
filename = r"Fri Aug 26 131517 CDT 2022"
            
#Load in the data. This is a hashmap with the device name as the key and the 
#BioSignal/event markers as the value.
sessionData = sigimport.importTattooData(path, filename)

#%% Showing how to get signals from the session data
#Data and markers from the first device (PPG Tattoo V3.2)
eegSignals = sessionData["EEG-Test"]["Data"]
# ppgMarkers = sessionData["PPG Tattoo v3.2"]["Markers"]

#Data and markers from the second device (Pneumonia_v2.0)
# ecgSignals = sessionData["Pneumonia_v2.0"]["Data"]
# ecgMarkers = sessionData["Pneumonia_v2.0"]["Markers"]

#%% Example signal processing on the PPG signals
signals = copy.deepcopy(eegSignals)
# markers = copy.deepcopy(eegMarkers)
markers = []

#Here are some examples of basic signal processing of Biosignals.

#Ex 1. Convert byte data of temperature sensor into celsius.
# signals[11].fs = 25/8
# signals[11].convert_TMP117()

#Ex 2. Apply a bandpass filter to the PPG channels
num_ppg_channels = 8

for i in range(0, num_ppg_channels):
   signals[i].filter_signal("Bandpass", [1, 50])

# Ex 3. Sometimes the tattoo disconnects, and no data was recorded. We can pad
# the signals to maintain proper timing of the signals.
for i in range(0, len(signals)):
    markers_pad = signals[i].pad_disconnect_events(markers, method = "hold")

#%% Plot data
plt.close('all')
ch_select = [1, 2, 3, 4, 6, 7]
sigplot.plot_biosignals([signals[i] for i in ch_select], markers)

#Plotting just a single channel
# signals[2].plot(markers)
plt.figure()
signals[2].plot_stft()
plt.title(signals[2].name)
          
# Simple check for channel redundancies 
d = {'Ch2': signals[2].data, 'Ch3': signals[3].data, 'Ch4': signals[4].data, 'Ch5': signals[5].data}
df = pd.DataFrame(d)
print('Pearson correlation coefficients')
print(df.corr(method='pearson', min_periods=1))

# You can access the time series data as a Numpy array as well
ppg1_np_array = signals[0].data #Retrieving the Biosignals' time series data as a Numpy array
ppg1_t = np.linspace(0, len(ppg1_np_array) / signals[0].fs, len(ppg1_np_array))
print([signals[i].fs for i in range(0,len(signals))])
plt.figure()
plt.plot(ppg1_t, ppg1_np_array)
plt.xlabel("t [s]")
plt.ylabel("ADC count [n]")
plt.title("Plotting with Numpy Arrays")


data_np_array = np.empty(shape=(signals[1].data.shape[0], len(ch_select)+1))
for i, ch in enumerate(ch_select):
    ch_np_array = signals[ch].data
    data_np_array[:, i+1] = ch_np_array

ch_t = np.linspace(0, len(ch_np_array) / signals[ch].fs, len(ch_np_array))
data_np_array[:, 0] = ch_t
np.savetxt(filename + ".csv", data_np_array, delimiter=",")