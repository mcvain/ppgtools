from scipy.signal import find_peaks
import numpy as np
from ppgtools import sigpro

import matplotlib.pyplot as plt

import heartpy as hp
from ecgdetectors import Detectors

def find_peaks_heartpy(inp, fs, plot = False):
    working_data, measures = hp.process(inp, fs)

def find_peaks_adaptive_threshold(inp, N):
    '''
    Function to determine the location of peaks using an adaptive threshold.
    The adaptive threshold is obtained from the moving average of the original signal.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks in.
    N : int
        Moving average size.

    Returns
    -------
    peaks : array_like
        Indices of the estimated peaks.

    '''
    peaks = []
    
    thresh = sigpro.moving_average(inp, N)
    intersections = np.argwhere(np.diff(np.sign(inp[0:-N+1] - thresh))).flatten()
    
    #Determine if the first crossing is positive or negative (i.e., does the signal start above or below the threshold)
    starts_positive = thresh[0] > inp[0]
    
    if starts_positive:
        i = 1
    else:
        i = 2
    while(i < len(intersections)):
        local_peak = np.argmax(inp[intersections[i-1]:intersections[i]])
        peaks.append(local_peak + intersections[i-1])
        i+=2
    
    peaks = np.asarray(peaks)
    return peaks

#####################################
# Peak detection using ecgdetectors #    
#####################################
def find_peaks_hamilton(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of P.S. Hamilton, “Open Source ECG Analysis Software Documentation”, E.P.Limited, 2002.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    detectors = Detectors(fs)
    return detectors.hamilton_detector(inp)
    
def find_peaks_christov(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of Ivaylo I. Christov, “Real time electrocardiogram QRS detection using combined adaptive threshold”, BioMedical Engineering OnLine 2004, vol. 3:28, 2004. 

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    detectors = Detectors(fs)
    return detectors.christov_detector(inp)

def find_peaks_engzee(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of W. Engelse and C. Zeelenberg, “A single scan algorithm for QRS detection and feature extraction”, IEEE Comp. in Cardiology, vol. 6, pp. 37-42, 1979 with modifications A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, “Real Time Electrocardiogram Segmentation for Finger Based ECG Biometrics”, BIOSIGNALS 2012, pp. 49-54, 2012.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    detectors = Detectors(fs)
    return detectors.engzee_detector(inp)

def find_peaks_pan_tompkins(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of Jiapu Pan and Willis J. Tompkins. “A Real-Time QRS Detection Algorithm”. In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    
    detectors = Detectors(fs)
    return detectors.pan_tompkins_detector(inp)

def find_peaks_swt(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation based on Vignesh Kalidas and Lakshman Tamil. “Real-time QRS detector using Stationary Wavelet Transform for Automated ECG Analysis”. In: 2017 IEEE 17th International Conference on Bioinformatics and Bioengineering (BIBE). Uses the Pan and Tompkins thresolding method.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    
    detectors = Detectors(fs)
    return detectors.swt_detector(inp)

def find_peaks_two_average(inp, fs):
    '''
    R Peak detector from the ecgdetectors package. 
    Implementation of Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). “Frequency Bands Effects on QRS Detection” The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010). 428-431.

    Parameters
    ----------
    inp : array_like
        Signal to find peaks.
    fs : int
        Sample rate.

    Returns
    -------
    array_like
        Time locations of the peaks.

    '''
    
    detectors = Detectors(fs)
    return detectors.two_average_detector(inp)
