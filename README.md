# ppgtools
Python package for parsing and processing .bin data from Pulse

**Data format**
Pulse saves data in a .tat file. The .tat file starts off with a header that lists the relevant parameters of each channel of data, such as sample rate, bytes per point, etc. After the header, the .tat file contains 
You can use the module "sigimport.importTAT(" to parse the .tat file. It will return a dict, with key-value pairs for each BLE device that was connected. Each dict value is another dict containing a list of Biosignal objects and the event markers for that particular BLE device.

**Biosignal Object**
The Biosignal object is just a data structure that packages the sensor time series data with important parameters such as its name, sample rate, and units. It also has some functions to manipulate the time series data, such as filtering, resampling, and trimming.

![image](https://user-images.githubusercontent.com/19411810/160197212-34cb752d-2f02-48b8-8a6e-9ebc798605f9.png)


**Dependencies**
You will need Numpy, SciPy, HeartPy, and py-ecg-detectors

**Getting Started**
There is an example script "Example_Loading_Tat_Data.py" that can help walk through importing tattoo data and basic manipulation and plotting.

**Modules**
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
