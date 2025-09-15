from matplotlib import pyplot as plt
import numpy as np
from ppgtools import sigplot
from ppgtools.sigimport import importTattooData

tattoo_path = "data"
tattoo_filename = "Mon Feb 12 093349 CST 2024 PV Chest"
numdata_ds_method = "true" # "true" or "naive"
notification_freq = 6.25

# load data
sessionData, devices = importTattooData(tattoo_path, tattoo_filename)
signals = sessionData[devices[0]]["Data"]
markers = sessionData[devices[0]]["Markers"]
signal_names = [s.name for s in signals]
print(f"Found {signal_names}")
for i in range(0, len(signals)): #Pad the signals as required for disconnections
    markers_pad = signals[i].pad_disconnect_events_new(markers, method = "hold")
user_markers = [m for m in markers_pad if not m.label.startswith("Sync time: ")]

# plot data
sigplot.plot_biosignals(signals[0:len(signals)], user_markers)
plt.show(block=False)

# each channel has some information available:
print(signals[0].name) # channel name
print(signals[0].fs) # sampling rate
print(signals[0].units) # data units
print(signals[0].data.shape) # data shape

# ADC conversion for TMP117 
for signal in signals:
    if signal.name == "TEMP":
        signal = signal.convert_TMP117()

# Some signals have known units 
for signal in signals:
    if signal.name in ["SV"]:
        signal.units = "mL"
    if signal.name in ["SBP", "DBP"]:
        signal.units = "mmHg"

### Post-hoc correction of sampling frequency is good to do since Pulse can only handle integer fs. 
# PPG: 31 Hz -> 31.25 Hz.
# Packet number: 6.25 Hz, matching the notification frequency of 6.25 Hz, aka how often send_all_data() is called in the firmware. 
# MC3635: 31 Hz -> 31.25 Hz.

for signal in signals:
    if signal.name in ["PPG"]:
        signal.fs = 125  # true with firmware written in NCS
    if signal.name in ["num"]:
        signal.fs = notification_freq
    print(f"Set sampling frequency for {signal.name} to {signal.fs} Hz")

# For timer-interrupt based sensors or machine learning inferred values, a value is being included in every packet, but this value is not always a true fresh sample from the sensor. The latest available value is simply filled into the packet.
# e.g. TEMP is sampled at 1 Hz, but the data is sent at 6.25 Hz.
# e.g. SV, SBP, DBP are inferred (i.e. "sampled") at 1/1.92 Hz, but the data is sent at 6.25 Hz.
# Here we have two options to deal with this:

# Method 1 (naive, better for visualization): simply treat them as 6.25 Hz signals, but keep in mind these are technically upsampled by the e-tattoo (method of upsampling: the latest available value is repeated until the next fresh sample is available).
if numdata_ds_method == 'naive':
    for signal in signals:
        if signal.name == "TEMP":
            signal.fs = notification_freq # Set the sampling frequency for TEMP

        if signal.name in ["SV", "SBP", "DBP"]:
            signal.fs = notification_freq
        print(f"Set sampling frequency for {signal.name} to {signal.fs} Hz")

# Method 2 (more accurate for analysis): mark the 1st "valid-DWN-packet" with a DWN value of 0, make this packet number 0, and make the analysis such that only every 12th packet contains a freshly sampled DWN value
if numdata_ds_method == 'true':
    for signal in signals:
        if signal.name in ["SV", "SBP", "DBP"]:
            fs_true = notification_freq / 12  # 0.5208... Hz
            data = np.array(signal.data)
            # Find the first non-zero index
            nonzero_indices = np.flatnonzero(data)
            if nonzero_indices.size > 0:
                first_idx = nonzero_indices[0]
                # Take every 12th sample starting from the first non-zero (i.e., 0, 12, 24, ...)
                sampled_data = data[first_idx::12]
                signal.data = sampled_data
                signal.fs = fs_true
            print(f"Set sampling frequency for {signal.name} to {signal.fs} Hz")
        else:
            pass

for signal in signals:
    print(f"{signal.name}: {signal.data.shape} samples, fs={signal.fs} Hz")
f, axes = sigplot.plot_biosignals(signals[0:len(signals)])  # , user_markers
for i, signal in enumerate(signals):
    if signal.name not in ["TEMP", "SV", "SBP", "DBP"]:
        axes[i].set_yticks([])
        axes[i].set_ylabel("")
    axes[i].set_xlim(60, 80)
f.set_figheight(f.get_figheight() * 2.5)
plt.tight_layout()
plt.show(block=True) 