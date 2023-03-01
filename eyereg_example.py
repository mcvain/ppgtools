import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from ppgtools import sigpro, sigimport, sigpeaks, sigplot, save, biometrics, sigseg, sigrecover
import copy
import mne 
from eogkill import eyereg
from math import ceil, floor 

# Load in files 
path = r"C:/Users/mcvai/ppgtools/data/eeg/USAARL Feb 2023 Shipping"
print(os.getcwd())
filename = r"Sat Feb 04 174357 CST 2023 board1_etat_artifacts_S1"

# Load in the data. This is a hashmap with the device name as the key and the 
# BioSignal/event markers as the value.
sessionData = sigimport.importTattooData(path, filename)

# Initialize some parameters while verifying the data source as the EEG board 
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
        markers = []
        board_id = released_6ch_boards.index(device) + 1
        num_channels = len(signals) - 3 # last channel contains sample # and not data. Further remove 2 for 2 blank channels 
chdata_list = []  
for i in range(0, num_channels):
    chdata_list.append(signals[i].data)
chdata_array = np.array(chdata_list)

# Apply a bandpass filter
for i in range(0, num_channels):
   signals[i].filter_signal("Bandpass", [1, 40])

# Sometimes the tattoo disconnects, and no data was recorded. We can pad
# the signals to maintain proper timing of the signals.
for i in range(0, len(signals)):
    markers_pad = signals[i].pad_disconnect_events(markers, method = "hold")
 
# Pack the data into mne format 
# ch_names = ["AFp7", "Fp1", "Fp1h", "Fp2h", "Fp2", "AFp8", "EOGh", "EOGv"] # for the eventual 8 channel board
ch_names = ["F7", "Fp1", "Fp2", "F8", "EOGh", "EOGv"] # for the 6 channel board
# or, if using simple channel names: 
# ch_names = [o.name for o in eegSignals]
# ch_names.remove('Packet')

chdata_array = chdata_array * 1.73846881 * pow(10, -8) # ADC_Value*(Vref/gain)/2^24

# Plot before correction 
ch_types = ["eeg","eeg","eeg","eeg","eog","eog"]
raw = mne.io.RawArray(chdata_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw_bandpass = raw.copy().filter(l_freq = 1, h_freq = 50)

my_annot = mne.Annotations(onset=[77.442, 80.190, 83.097, 85.473, 94.088, 97.387, 100.365, 103.136, 3.254, 6.396, 8.317, 10.678],  # in seconds
                           duration=[2.059, 2.721, 2.04, 2.157, 2.307, 1.571, 1.715, 1.764, 0.485, 0.735, 1.528, 1.671],  # in seconds, too
                           description=['up', 'down', 'left', 'right', 'up', 'down', 'left', 'right', 'blink', 'blink', 'blink', 'blink'])
raw_bandpass.set_annotations(my_annot)
fig = raw_bandpass.plot(scalings=dict(eeg=20e-6, eog=20e-6), show=True, block=False, remove_dc=True)
print(raw_bandpass.info)

# after plotting, switch to numerical labels 
description = raw_bandpass.annotations.description
new_description = []
for i in range(len(description)):
    if description[i] == 'up':
        new_description.append('3')
    elif description[i] == 'down':
        new_description.append('4')
    elif description[i] == 'left':
        new_description.append('2')
    elif description[i] == 'right':
        new_description.append('1')
    elif description[i] == 'blink':
        new_description.append('5')
raw_bandpass.annotations.description = new_description

# convert to label array 
fs = eegSignals[0].fs
labels_array = np.ones((1, np.size(chdata_array, 1))) * 6  # non-eye-movement i.e. resting should be 6 by default 
for ann in raw_bandpass.annotations:
    descr = ann["description"]
    start = floor(fs * ann["onset"]) # convert to samples
    end = ceil(fs * (ann["onset"] + ann["duration"])) # convert to samples
    labels_array[0, start:end] = float(descr)

# raw_bandpass.compute_psd(fmax=50, average='mean').plot()

eeg_chan_idxs = [i for i, e in enumerate(ch_types) if e == "eeg"]
eog_chan_idxs = [i for i, e in enumerate(ch_types) if e == "eog"]

eyereg_inst = eyereg()
eyereg_inst.fit(chdata_array, labels_array, eeg_chan_idxs, eog_chan_idxs)

# Plot after correction
# bring in new data 
filename = r"Sat Feb 04 180213 CST 2023 board1_etat_nback2_S1"
sessionData = sigimport.importTattooData(path, filename)

# Initialize some parameters while verifying the data source as the EEG board 
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
        markers = []
        board_id = released_6ch_boards.index(device) + 1
        num_channels = len(signals) - 3 # last channel contains sample # and not data. Further remove 2 for 2 blank channels 
chdata_list = []  
for i in range(0, num_channels):
    chdata_list.append(signals[i].data)
chdata_array = np.array(chdata_list)
chdata_array = chdata_array * 1.73846881 * pow(10, -8) # ADC_Value*(Vref/gain)/2^24

raw = mne.io.RawArray(chdata_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw_bandpass = raw.copy().filter(l_freq = 1, h_freq = 50)
fig = raw_bandpass.plot(scalings=dict(eeg=20e-6, eog=20e-6), show=True, block=False, remove_dc=True, title="Before EYEREG")

corrected = eyereg_inst.apply(chdata_array)
corrected = mne.io.RawArray(corrected, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
corrected_bandpass = corrected.copy().filter(l_freq = 1, h_freq = 50)
fig = corrected_bandpass.plot(scalings=dict(eeg=20e-6, eog=20e-6), show=True, block=True, remove_dc=True, title="After EYEREG")
mne.preprocessing.EOGRegression()