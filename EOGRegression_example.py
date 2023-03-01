import os
import numpy as np 
from ppgtools import sigpro, sigimport, sigpeaks, sigplot, save, biometrics, sigseg, sigrecover
import copy
import mne  

# Load in files 
path = r"C:/Users/mcvai/ppgtools/data/eeg/USAARL Feb 2023 Shipping"
print(os.getcwd())
filename = r"Sat Feb 04 174357 CST 2023 board1_etat_artifacts_S1"

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

ch_types = ["eeg","eeg","eeg","eeg","eog","eog"]
raw = mne.io.RawArray(chdata_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw.load_data()

raw.set_eeg_reference('average')
raw.filter(0.3, 40)
my_annot = mne.Annotations(onset=[77.442, 80.190, 83.097, 85.473, 94.088, 97.387, 100.365, 103.136, 3.254, 6.396, 8.317, 10.678],  # in seconds
                           duration=[2.059, 2.721, 2.04, 2.157, 2.307, 1.571, 1.715, 1.764, 0.485, 0.735, 1.528, 1.671],  # in seconds, too
                           description=['up', 'down', 'left', 'right', 'up', 'down', 'left', 'right', 'blink', 'blink', 'blink', 'blink'])
raw.set_annotations(my_annot)

# 1. automatic blink detection
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_evoked = eog_epochs.average('all')
fig = eog_evoked.plot('all', window_title='automatic EOG event detection: (max(eog) - min(eog)) / 4')

# 2. from manually annotated EOG
# events, event_id = mne.annotations.events_from_annotations(raw)
# eog_epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.5, tmax=0.5, baseline=(-0.5, -0.2))
# eog_evoked = eog_epochs.average('all')
# fig = eog_evoked.plot('all', window_title='manually annotated EOG event detection')

# perform regression on the evoked blink response
model_evoked = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(eog_evoked)
# fig = model_evoked.plot(vlim=(None, 0.4))
# fig.set_size_inches(3, 2)

# for good measure, also show the effect on the blink evoked
eog_evoked_clean = model_evoked.apply(eog_evoked)
eog_evoked_clean.apply_baseline((-0.5, -0.2))
fig = eog_evoked_clean.plot('all', window_title='effect of correction on blink events')

# apply the regression coefficients to the original data
fig = raw.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=False, remove_dc=True, title="Before EOGRegression")
raw_clean = model_evoked.apply(raw)
fig = raw_clean.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=True, remove_dc=True, title="After EOGRegression")
psd = raw.compute_psd(fmax=50).plot(show=True)
psd = raw_clean.compute_psd(fmax=50).plot(show=True)
print("Done!")