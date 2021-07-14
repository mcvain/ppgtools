import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cmath
import statsmodels.api as sm
from ppgtools import sigpeaks, sigpro, sigplot
import ppgtools
import copy

#Constants
col_width = 16
calibration_fs = 20

#Options for what type of algorithm will be used
use_xcorr = False
use_phase_slope = True



class Delay_Extractor():
    #Debug text
    show_delays = False
    show_xcorr = False
    show_phase_slope = False

    #Initialize a Delay_Extractor object. Here, we can set debug text, etc.
    def __init__(self):
        pass
    
    def delay_crosscorrelation(self, proximal_signal, distal_signal):
        pass
    
    def delay_phase_slope(self, proximal_signal, distal_signal):
        pass
    
    def delay_sdm(self, proximal_signal, distal_signal):
        pass
    
    def delay_peaks(self, proximal_signal, distal_signal):
        #Make a deep copy
        prox_sig = copy.deepcopy(proximal_signal)
        dist_sig = copy.deepcopy(distal_signal)
        
        #Overfilter
        prox_sig.filter_signal("bandpass", [0.5, 3])
        dist_sig.filter_signal("bandpass", [0.5, 3])
        
        #Find peaks
        prox_peaks = prox_sig.find_peaks_adaptive_threshold(2)
        dist_peaks = dist_sig.find_peaks_adaptive_threshold(2)
        
        sigplot.plot_biosignals([prox_sig, dist_sig], peaks = [prox_peaks, dist_peaks])
        
        
        #Find the time difference between the two peaks
        ptt = np.subtract(dist_peaks, prox_peaks)
        
        #Convert to seconds
        ptt = ptt / prox_sig.fs
        
        return prox_peaks, dist_peaks, ptt
    