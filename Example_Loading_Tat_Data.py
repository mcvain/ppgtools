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
from asrpy import *
from scipy.signal import find_peaks
from math import log10
import mne
from mne.preprocessing import EOGRegression

#Load in ppgtools package.
#You may have to add the relative location of the package. Any better way?
#sys.path.insert(0, r'../../../ppgtools/') 
from ppgtools import sigpro, sigimport, sigpeaks, sigplot, save, biometrics, sigseg, sigrecover


#%% Load in files.
path = r"C:/Users/mcvai/ppgtools/data/eeg/USAARL Feb 2023 Shipping"
print(os.getcwd())
filename = r"Sat Feb 04 180213 CST 2023 board1_etat_nback2_S1"

#Load in the data. This is a hashmap with the device name as the key and the 
#BioSignal/event markers as the value.
sessionData = sigimport.importTattooData(path, filename)

#%% Showing how to get signals from the session data
#Data and markers from the first device (PPG Tattoo V3.2)

released_6ch_boards = ["EEG-Test_E0.85.DB.5C.86.C0", "EEG-Test_C2.F7.64.EB.5B.6B"]
try:
    # old boards don't have ID in the filename 
    eegSignals = sessionData["EEG-Test"]["Data"]
except:
    assert len(sessionData.keys()) == 1
    device = list(sessionData.keys())[0]
    if device in released_6ch_boards:
        eegSignals = sessionData[device]["Data"]
        signals = copy.deepcopy(eegSignals)
        # markers = copy.deepcopy(eegMarkers)
        markers = []

        board_id = released_6ch_boards.index(device) + 1
        num_channels = len(signals) - 3 # last channel contains sample # and not data. Further remove 2 for 2 blank channels 

# ppgMarkers = sessionData["PPG Tattoo v3.2"]["Markers"]

#%% Example signal processing on the PPG signals
#Here are some examples of basic signal processing of Biosignals.

#Ex 1. Convert byte data of temperature sensor into celsius.
# signals[11].fs = 25/8
# signals[11].convert_TMP117()

#Ex 2. Apply a bandpass filter to the PPG channels
# num_ppg_channels = 8
chdata_list = []   
for i in range(0, num_channels):
    chdata_list.append(signals[i].data)

chdata_array = np.array(chdata_list)

# (optional) make sure your asr is only fitted to clean parts of the data
# pre_cleaned, _ = clean_windows(chdata_array, signals[0].fs, max_bad_chans=0.1)

# fit the asr
# M, T = asr_calibrate(pre_cleaned, signals[0].fs, cutoff=15)

# apply it
# clean_array = asr_process(chdata_array, signals[0].fs, M, T)

# plot 
# sigplot.plot_biosignals(chdata_array, markers)

for i in range(0, num_channels):
   signals[i].filter_signal("Bandpass", [1, 40])

# Ex 3. Sometimes the tattoo disconnects, and no data was recorded. We can pad
# the signals to maintain proper timing of the signals.
for i in range(0, len(signals)):
    markers_pad = signals[i].pad_disconnect_events(markers, method = "hold")

# Data export to mne
# ch_names = ["AFp7", "Fp1", "Fp1h", "Fp2h", "Fp2", "AFp8", "EOGh", "EOGv"]
ch_names = ["F7", "Fp1", "Fp2", "F8", "EOGh", "EOGv"]

# or: 
#ch_names = [o.name for o in eegSignals]
#ch_names.remove('Packet')

# chdata_array = chdata_array * (4.5/24) * (1/pow(2, 24))  # ADC_Value*(Vref/gain)/2^24
chdata_array = chdata_array * 1.73846881 * pow(10, -8)

raw = mne.io.RawArray(chdata_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=["eeg","eeg","eeg","eeg","eog","eog"]))
raw_bandpass = raw.copy().filter(l_freq = 1, h_freq = 50)
raw_bandpass.plot(scalings='auto', show=True, block=False, remove_dc=True)
print(raw_bandpass.info)
raw_bandpass.plot_psd(fmax=50, average=True)

raw_bandpass = raw_bandpass.copy().set_eeg_reference(ref_channels='average')
EOG_corr_weights = EOGRegression(picks='eeg', picks_artifact='eog', proj=False).fit(raw_bandpass)
corrected = EOG_corr_weights.apply(raw_bandpass, copy=True)
corrected.plot(scalings='auto', show=True, block=True, remove_dc=True)

#%% Plot data
plt.close('all')
ch_select = range(0, num_channels)
sigplot.plot_biosignals([signals[i] for i in ch_select], markers)

#Plotting just a single channel
# signals[2].plot(markers)
plt.figure()
signals[1].plot_stft(win_len=5)
plt.show(block=False)

# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
plt.figure()

for ch in range(num_channels):
    pxx, freq = plt.psd(signals[ch].data, NFFT=256, Fs=signals[0].fs, detrend='none', sides='onesided', scale_by_freq=True, label=str(signals[ch].name))
plt.legend()
# peakso, properties = find_peaks(pxx, threshold=45000)
# freq_peaks = [] 
# peaks = peakso.tolist()
# print(peaks)
# for p in peaks:
#     freq_peaks.append(freq[p])
# print(str(freq_peaks) + " Hz; ")
# plt.plot(peakso, 10*np.log10(pxx[peakso]), "x")
plt.xlim([0, 50])
plt.savefig('C:/Users/mcvai/EEG-FPC/v1/debug-notes/' + str(filename), dpi=300)
plt.show()

# Simple check for channel redundancies / cross-talk / whatever 
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