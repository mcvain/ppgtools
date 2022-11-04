# ppgtools
Python package for parsing and processing .bin data from Pulse

### Dependencies
You will need Numpy, SciPy, HeartPy, and py-ecg-detectors

### Getting Started 
Pulse app will output a folder which is timestamped. Copy the whole folder into `path`, which is set to `data/eeg` by default.

You can use "Example_Loading_Tat_Data.py", but replace the string in line:  
`filename = r"Fri Aug 26 131517 CDT 2022"`  
with your folder name.  

### Modules
- biometrics.py   Module that contains basic biometric extraction algorithms (HR, SpO2, RR, etc.)
- biosignal.py    Defines the Biosignal object
- save.py         Module that contains functions to save and load biometric data
- sigdelay.py     Module that contains algorithms for time delay (xcorr, intersecting tangent, phase slope, SDM, peak-to-peak)
- sigimport.py    Module for loading in data from files (tattoo data, BioPac data, Wellue data, etc.)
- sigpeaks.py     Module for finding the peaks in data (adaptive threshold, SDM, local mins, intersecting tangent, pan-tompkins, etc.)
- sigplot.py      Module for plotting Biosignals
- sigpro.py       Module for manipulating time series data (Numpy arrays)
- sigrecover.py   Module for recovering noisy data (under work, only ICA)
- sigseg.py       Module for determining and rejecting noisy segments in data
