from matplotlib import pyplot as plt
import numpy as np
from ppgtools import sigplot
from ppgtools.sigimport import importTattooData

tattoo_path = "data"
tattoo_filename = "Tue Feb 13 070135 CST 2024 PV Hydration"
numdata_ds_method = "true" # "true" or "naive"
notification_freq = 1

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

for signal in signals:
    print(f"{signal.name}: {signal.data.shape} samples, fs={signal.fs} Hz")
f, axes = sigplot.plot_biosignals(signals[0:len(signals)])  # , user_markers
for i, signal in enumerate(signals):
    if signal.name not in ["TEMP", "SV", "SBP", "DBP"]:
        axes[i].set_yticks([])
        axes[i].set_ylabel("")
f.set_figheight(f.get_figheight() * 2.5)
plt.tight_layout()
plt.show(block=True) 