import os
import numpy as np 
from ppgtools import sigpro, sigimport, sigpeaks, sigplot, save, biometrics, sigseg, sigrecover
import copy
import mne  
from matplotlib import pyplot as plt 
import matplotlib.animation as animation
# plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin'

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
model_evoked = mne.preprocessing.EOGRegression(picks='eeg', picks_artifact='eog', proj=True).fit(eog_evoked)

fig = raw.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=False, remove_dc=True, title="before (training)")
raw_clean = model_evoked.apply(raw)
fig = raw_clean.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=False, remove_dc=True, title="after (training)")

# On to testing on unseen data 
raw = mne.io.RawArray(test_data_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw.load_data()
raw.filter(0.3, 40)
fig = raw.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=False, remove_dc=True, title="before (testing)")

cnt = 5000 # starting sample 
original_test_data_array = np.zeros((7, test_data_array.shape[1]))
corrected_test_data_array = np.zeros((7, test_data_array.shape[1]))
t = 0
while cnt+1 < test_data_array.shape[1]:
    cnt += 1 # sample
    incoming_data = test_data_array[:, cnt]
    corrected_data = incoming_data.copy()
    corrected_data[0:4] = incoming_data[0:4] - model_evoked.coef_ @ incoming_data[4:6]
    test_data_array[:, cnt] = corrected_data

    t += 1 / eegSignals[0].fs
    corrected_test_data_array[0, cnt] = t
    corrected_test_data_array[1:, cnt] = corrected_data

    original_test_data_array[0, cnt] = t
    original_test_data_array[1:, cnt] = incoming_data

raw = mne.io.RawArray(test_data_array, mne.create_info(ch_names, eegSignals[0].fs, ch_types=ch_types))
raw.load_data()
raw.filter(0.3, 40)
fig = raw.plot(scalings=dict(eeg=20e-6, eog=150e-6), show=True, block=False, remove_dc=True, title="after (testing)")

# Animation of pre-generated before and after data 
fig, ax = plt.subplots(6, 2, figsize=(8, 12))
fig.tight_layout(pad=3.0)

# Define the animation update function for each subplot
def update(i):
    # Get the x-axis and y-axis data for the current frame
    x = original_test_data_array[0, :i]
    y = original_test_data_array[1:, :i]
    y_corr = corrected_test_data_array[1:, :i]
    
    lines = []
    for j, subplot in enumerate(ax.flat):

        if j in [0, 2, 4, 6, 8, 10]:
            # Update xlim
            if i > 2500+5001 and i % 2500+5001 == 0:
                subplot.set_xlim((x[i], x[i+2500+5001]))
            
            # Update ylim
            start_idx = max(0, i-5000)
            end_idx = i
            # subplot.set_ylim((np.percentile(y[int(j/2), :], 3), np.percentile(y[int(j/2), :], 97)))
            subplot.set_ylim((-0.005, 0.005))
            
            line, = subplot.plot(x, y[int(j/2)], color='blue')
        else: # [1, 3, 5, 7, 9, 11] 
            # Update xlim
            if i > 2500+5001 and i % 2500+5001 == 0:
                subplot.set_xlim((x[i], x[i+2500+5001]))
            
            # Update ylim
            start_idx = max(0, i-5000)
            end_idx = i
            # subplot.set_ylim((np.percentile(y_corr[int((j-1)/2), :], 3), np.percentile(y_corr[int((j-1)/2), :], 97)))
            subplot.set_ylim((-0.005, 0.005))

            line, = subplot.plot(x, y_corr[int((j-1)/2)], color='red')
        lines.append(line)
        subplot.set_title(f'Subplot {j+1}')
        #subplot.set_xlim(0, np.max(test_data_array[0,:]))
        #subplot.set_ylim(0, np.max(test_data_array[1:,:]))

    label = ax[0,0].text(0, 0, str(i), ha='center', va='center', fontsize=20, color="Red")
    # Return the updated lines
    lines.append(label)
    return lines

# Create the animation for each subplot
# animation.FFMpegWriter.bin_path = 'C:/ffmpeg/bin'

ani = animation.FuncAnimation(fig, update, frames=range(5001, test_data_array.shape[1]), interval=1, blit=True, repeat=False)

# f = r"animation.gif" 
# writergif = animation.PillowWriter(fps=30) 
# ani.save(f, writer=writergif)
# ani.save('animation.mp4', writer=writer)
# Show the plot
plt.show()