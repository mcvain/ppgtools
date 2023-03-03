import os
import numpy as np 
from ppgtools import sigpro, sigimport, sigpeaks, sigplot, save, biometrics, sigseg, sigrecover
import copy
import mne  
from matplotlib import pyplot as plt 
import matplotlib.animation as animation
from scipy.stats import zscore
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def raw_custom_plot(raw, fig, ax, ax_idxs, times, mean=False, events=[]):
    # channels to plot:
    to_plot = ["eeg"]

    # get the data for plotting in a short time interval from 2 to 22 seconds
    start = int(raw.info['sfreq'] * times[0])
    stop = int(raw.info['sfreq'] * times[1])
    data, times = raw.get_data(picks=to_plot,
                            start=start, stop=stop, return_times=True)

    # Scale the data from the MNE internal unit V to µV
    # data *= 1e6
    # data = zscore(data, axis=1)
    data = butter_bandpass_filter(data, 2, 40, 250)
    
    ax[ax_idxs[0], ax_idxs[1]].cla()
    if mean:
        # Take the mean of the channels
        data_mean = np.mean(data, axis=0)

        # plot some EEG data
        ax[ax_idxs[0], ax_idxs[1]].plot(times, data_mean)
    else:
        print(times)
        print(data.T)
        ax[ax_idxs[0], ax_idxs[1]].plot(times.T, data.T, linewidth=0.5)
    ax[ax_idxs[0], ax_idxs[1]].set_title('EEG data')
    ax[ax_idxs[0], ax_idxs[1]].set_xlabel('Time (s)')
    ax[ax_idxs[0], ax_idxs[1]].set_xlim(times[0], times[-1])

    ax[ax_idxs[0], ax_idxs[1]].set_ylim(-4e-05, 4e-05)
    ax[ax_idxs[0], ax_idxs[1]].set_ylabel(u'Amplitude (\u03bcV)')

    events = mne.annotations_from_events(events, 250).onset
    events = events.tolist()
    for ev in events:
        ax[ax_idxs[0], ax_idxs[1]].vlines(ev, -99999, 99999, alpha=0.15, color='black', linewidth=0.8)

    return fig, ax

# Load in training file 
path = r"C:/Users/mcvai/ppgtools/data/eeg/USAARL Feb 2023 Shipping"
print(os.getcwd())
# filename = r"Sat Feb 04 174357 CST 2023 board1_etat_artifacts_S1"
filename = r"Fri Feb 03 174308 CST 2023 board1_gel_artifacts_S1"
# filename = r"Sat Feb 04 163920 CST 2023 board2_gel_artifacts_S1"

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
training_data_array = np.array(chdata_list)

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

training_data_array = training_data_array * 1.73846881 * pow(10, -8) # ADC_Value*(Vref/gain)/2^24
ch_types = ["eeg","eeg","eeg","eeg","eog","eog"]

## Test set  
# filename = r"Sat Feb 04 174651 CST 2023 board1_etat_nback1_S1"
filename = r"Fri Feb 03 172818 CST 2023 board1_gel_nback1_S1"
# filename = r"Fri Feb 03 174308 CST 2023 board1_gel_artifacts_S1"
sessionData = sigimport.importTattooData(path, filename)

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
test_data_array = np.array(chdata_list)

test_data_array = test_data_array * 1.73846881 * pow(10, -8) # ADC_Value*(Vref/gain)/2^24

raw = mne.io.RawArray(training_data_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw.load_data()
raw.set_eeg_reference('average')
raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

mfig, ax = plt.subplots(nrows=6, ncols=2, figsize=(10, 10),
                       sharex='row', sharey='row')
mfig.subplots_adjust(hspace=0.8)

# automatic blink detection with create_eog_epochs
train_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-1, -0.5), tmin=-1, tmax=0.5)
train_evoked = train_epochs.average('all')
eog_events = mne.preprocessing.find_eog_events(raw, thresh=60e-6)
train_epochs.average('all').plot(axes=ax[0:2, 0], spatial_colors=True, selectable=False, show=False)

# perform regression on the evoked blink response
model_evoked = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(train_evoked)

raw.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=False, block=False, remove_dc=True, title="before (training)", highpass=0.3, lowpass=40, events=eog_events)
raw_clean = model_evoked.apply(raw)
raw_clean.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=False, block=False, remove_dc=True, title="after (training)", highpass=0.3, lowpass=40, events=eog_events)
mfig, ax = raw_custom_plot(raw, mfig, ax, [2, 0], times= [5,25], mean=False, events=eog_events)
mfig, ax = raw_custom_plot(raw_clean, mfig, ax, [2, 1], times= [5,25], mean=False, events=eog_events)

train_clean_evoked = mne.preprocessing.create_eog_epochs(raw_clean, baseline=(-1, -0.5), tmin=-1, tmax=0.5)
train_clean_evoked.average('all').plot(axes=ax[0:2, 1], spatial_colors=True, selectable=False, show=False)

raw = mne.io.RawArray(test_data_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw.load_data()
raw.set_eeg_reference('average')
raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

# test set data should be filtered because of the insane drifts
raw = raw.filter(2, 40)

test_evoked = mne.preprocessing.create_eog_epochs(raw, baseline=(-1, -0.5), tmin=-1, tmax=0.5)
eog_events = mne.preprocessing.find_eog_events(raw, thresh=35e-6)
test_evoked.average('all').plot(axes=ax[3:5, 0], spatial_colors=True, selectable=False, show=False)

fig = raw.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=False, block=False, remove_dc=True, title="before (testing)", highpass=0.3, lowpass=40, events=eog_events)
raw_clean = model_evoked.apply(raw)

test_clean_evoked = mne.preprocessing.create_eog_epochs(raw_clean, baseline=(-1, -0.5), tmin=-1, tmax=0.5)
test_clean_evoked.average('all').plot(axes=ax[3:5, 1], spatial_colors=True, selectable=False, show=False)

mfig, ax = raw_custom_plot(raw, mfig, ax, [5, 0], times= [398, 418], mean=False, events=eog_events)
mfig, ax = raw_custom_plot(raw_clean, mfig, ax, [5, 1], times= [398, 418], mean=False, events=eog_events)

mfig.show()

fig = raw_clean.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=True, remove_dc=True, title="after (testing)", highpass=0.3, lowpass=40, events=eog_events)

# mini-conclusion: doesn't work either. 
# so I think EOGRegression strictly works on training data, and not on test data; especially if test data is recorded with a different configuration. 
