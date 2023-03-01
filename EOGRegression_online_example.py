import os
import numpy as np 
from ppgtools import sigpro, sigimport, sigpeaks, sigplot, save, biometrics, sigseg, sigrecover
import copy
import mne  
from matplotlib import pyplot as plt 
import matplotlib.animation as animation

# Load in training file 
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
filename = r"Sat Feb 04 174651 CST 2023 board1_etat_nback1_S1"
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
raw.filter(0.3, 40)
raw.set_eeg_reference('average')
my_annot = mne.Annotations(onset=[77.442, 80.190, 83.097, 85.473, 94.088, 97.387, 100.365, 103.136, 3.254, 6.396, 8.317, 10.678],  # in seconds
                           duration=[2.059, 2.721, 2.04, 2.157, 2.307, 1.571, 1.715, 1.764, 0.485, 0.735, 1.528, 1.671],  # in seconds, too
                           description=['up', 'down', 'left', 'right', 'up', 'down', 'left', 'right', 'blink', 'blink', 'blink', 'blink'])
raw.set_annotations(my_annot)

# 1. automatic blink detection
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_evoked = eog_epochs.average('all')

# perform regression on the evoked blink response
model_evoked = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog').fit(eog_evoked)

raw = mne.io.RawArray(test_data_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw.load_data()

fig = raw.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=False, remove_dc=True, title="before")

def data_gen():
    t = data_gen.t
    cnt = 5000
    while cnt < len(test_data_array[0]):
        cnt+=1 # sample
        incoming_data = test_data_array[:, cnt]
        corrected_data = incoming_data.copy()
        corrected_data[0:4] = incoming_data[0:4] - model_evoked.coef_ @ incoming_data[4:6]
        test_data_array[:, cnt] = corrected_data

        t += 1 / eegSignals[0].fs

        yield t, corrected_data

data_gen.t = 0

raw = mne.io.RawArray(test_data_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw.load_data()
raw.filter(0.3, 40)
fig = raw.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=False, remove_dc=True, title="after")

def data_gen_original():
    t = data_gen_original.t
    cnt = 5000
    while cnt < len(test_data_array[0]):
        cnt+=1 # sample
        incoming_data = test_data_array[:, cnt]
        corrected_data = incoming_data.copy()
        corrected_data[0:4] = incoming_data[0:4] - model_evoked.coef_ @ incoming_data[4:6]
        test_data_array[:, cnt] = corrected_data

        t += 1 / eegSignals[0].fs

        yield t, incoming_data

data_gen_original.t = 0

fig, axes = plt.subplots(6, 2, figsize=(8, 12))
fig.tight_layout(pad=3.0)

line = []
for li, linen in enumerate(range(len(ch_names))):
    linen, = axes[li, 1].plot([], [], lw=2, color='r')
    # linen, = axes[li].plot([], [], lw=2, color='r')
    line.append(linen)

line_original = [] 
for li, linen in enumerate(range(len(ch_names))):
    linen, = axes[li, 0].plot([], [], lw=2, color='b')
    # linen, = axes[li].plot([], [], lw=2, color='b')
    line_original.append(linen)

# for ax in axes:
#     ax.grid()

# initialize the data arrays 
xdata, ydata = [], [[],[],[],[],[],[]]
# def run(data):
#     # update the data
#     t, ys = data
#     xdata.append(t)
#     for i, ch in enumerate(range(len(ch_names))):
#         ydata[i].append(ys[i])
#         # originalydata[i].append(original_ys[i])

#     # axis limits scrolling 
#     for i, ax in enumerate(axes[0:6,1]):
#     # for i, ax in enumerate(axes[6:12]):
#         xmin, xmax = ax.get_xlim()
#         if t >= xmax:
#             ax.set_xlim(xmax-10, 2*xmax)
#             ax.figure.canvas.draw()

#         # ax.set_ylim(min(ydata[i]), max(ydata[i])) #added ax attribute here
#         # ax.set_ylim(np.percentile(ydata[i][-2500:], 15), np.percentile(ydata[i][-2500:], 85))
#         # ax.set_ylim(np.median(ydata[i]) - 100 * pow(10, -6), np.median(ydata[i]) + 100 * pow(10, -6))
#         # ax.set_ylim(ydata[i][-1] - 100 * pow(10, -6), ydata[i][-1] + 100 * pow(10, -6))

#     for i, ch in enumerate(range(len(ch_names))):
#         line[i].set_data(xdata, ydata[i])

#     return line

def run(data):
    t, ys = data 
    xdata.append(t)
    for i, ch in enumerate(range(len(ch_names))):
        ydata[i].append(ys[i])
        
    for i in range(6): 
        line[i].set_data(xdata, ydata)
    return line + line_original

# xdata, yoriginaldata = [], [[],[],[],[],[],[]]
# def run_original(data):
#     # update the data
#     t, original_ys = data
#     xdata.append(t)
#     for i, ch in enumerate(range(len(ch_names))):
#         # ydata[i].append(ys[i])
#         yoriginaldata[i].append(original_ys[i])

#     # axis limits scrolling 
#     for i, ax in enumerate(axes[0:6,0]):
#     # for i, ax in enumerate(axes[0:6]):
#         xmin, xmax = ax.get_xlim()
#         if t >= xmax:
#             ax.set_xlim(xmax-10, 2*xmax)
#             ax.figure.canvas.draw()

#         # ax.set_ylim(np.percentile(yoriginaldata[i][-2500:], 15), np.percentile(yoriginaldata[i][-2500:], 85))

#     # update the data of both line objects
#     for i, ch in enumerate(range(len(ch_names))):
#         line_original[i].set_data(xdata, yoriginaldata[i])

#     return line_original
    
ani1 = animation.FuncAnimation(fig, run, blit=True, interval=10,
    repeat=False)
# ani2 = animation.FuncAnimation(fig, run_original, data_gen_original, blit=True, interval=10,
#     repeat=False)
plt.show()
