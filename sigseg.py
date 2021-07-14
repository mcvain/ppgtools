from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sigpro import find_all_peaks

col_width = 16

'''////////////////////Waveform Segmentation/////////////////////////'''
#Using foot of the wave, segment the signal and normalize each one.
def normalized_time_signals(sig, fs):
    window_size = 6 #s
    window_overlap = 3 #s

    #SDM, _, _ = find_peaks_and_mins(sig, fs, fs*window_size, window_overlap*fs, int(0.35 * fs) )
    #SDM = find_all_mins(sig, fs, fs*window_size, window_overlap*fs)
    SDM = find_all_peaks(sig, fs, fs*window_size, window_overlap*fs)
    #SDM = find_all_peaks(np.diff(np.diff(sig)), fs, fs*window_size, window_overlap*fs)
    
    plt.figure()
    plt.plot(np.linspace(0,len(sig)/fs,len(sig)), sig)
    
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
#    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2 = fig.add_axes([0.1, 0.5, 0.8, 0.4], sharex = ax1)
    ax1.plot(sig)
    ax1.plot(SDM, [sig[x] for x in SDM], "ro")

    ax2.plot(np.diff(np.diff(sig)) * 1000 * 1000)
    
    all_norm_sigs = []
    
    for i in range (1, len(SDM)):
        cur_wave = sig[SDM[i-1]:SDM[i]]
        if len(cur_wave > 0):
            all_norm_sigs.append(signal.resample(cur_wave, 1000))
        
    return all_norm_sigs

#Using an ecg signal, segment the signal and normalize each one.
def normalized_time_signals_ecg(sig, ecg, fs):
    window_size = 6 #s
    window_overlap = 3 #s

    #SDM, _, _ = find_peaks_and_mins(sig, fs, fs*window_size, window_overlap*fs, int(0.35 * fs) )
    r_peak = find_all_peaks(ecg, fs, fs*window_size, window_overlap*fs)
    #SDM = find_all_peaks(np.diff(np.diff(sig)), fs, fs*window_size, window_overlap*fs)
    
    plt.figure()
    plt.plot(np.linspace(0,len(sig)/fs,len(sig)), sig)
    
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
#    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax2 = fig.add_axes([0.1, 0.5, 0.8, 0.4], sharex = ax1)
    ax1.plot(ecg)
    ax1.plot(r_peak, [ecg[x] for x in r_peak], "ro")
    ax2.plot(np.diff(np.diff(sig)) * 1000 * 1000)
    
    all_norm_sigs = []
    
    for i in range (1, len(r_peak)):
        cur_wave = sig[r_peak[i-1]:r_peak[i]]
        if len(cur_wave > 0):
            all_norm_sigs.append(signal.resample(cur_wave, 1000))
        
    return all_norm_sigs    
    