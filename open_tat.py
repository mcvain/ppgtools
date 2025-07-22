from matplotlib import pyplot as plt
import numpy as np
from ppgtools import sigplot
from ppgtools.sigimport import importTattooData

tattoo_path = "data"
tattoo_filename = "example recording"

# load data
sessionData, devices = importTattooData(tattoo_path, tattoo_filename)
signals = sessionData[devices[0]]["Data"]
markers = sessionData[devices[0]]["Markers"]
print(f"Found {[s.name for s in signals]}")
for i in range(0, len(signals)): #Pad the signals as required for disconnections
    markers_pad = signals[i].pad_disconnect_events_new(markers, method = "hold")
user_markers = [m for m in markers_pad if not m.label.startswith("Sync time: ")]

# plot data
sigplot.plot_biosignals(signals[0:len(signals)], user_markers)
plt.show(block=False)

# each channel has some information available 
print(signals[0].name) # channel name
print(signals[0].fs) # sampling rate
print(signals[0].units) # data units
print(signals[0].data.shape) # data shape

X = signals[0].data # data as numpy array
t = np.linspace(0, len(X) / signals[0].fs, len(X))
plt.figure()
plt.title(signals[0].name)
plt.xlabel("Time (s)")
plt.ylabel(signals[0].units)
plt.plot(t, X)
plt.show(block=True)
