from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import bisect

col_width = 16

# =============================================================================
# 
# Toolbox of useful functions that can be used for processing PPG waveforms
# 1. Signal Conditioning
# 2. User Input
# 3. Plotting and Display
# 4. Waveform Segmentation
# 5. High Level Functions for Blood Pressure Estimation
# 6. Amplitude and Peak Extraction
# 7. Biometrics
# 8. Time Delay Extraction
# 
# =============================================================================
 
'''////////////////// Blood Pressure High Level Functions///////////'''
#Outputs PWV^2 * delta blood volume
def pivot_btb(t_amps, amps, t_pwv, pwv):
    out = []
    
    for i in range (0, len(t_amps)):
        upper_time = bisect.bisect_left(t_pwv, t_amps[i])
        
        if upper_time >= len(t_pwv):
            upper_time = len(t_pwv) - 1
        
        if upper_time < len(pwv):
            #Linearly interpolate to find the exact PWV value at the pulse time
            a = (pwv[upper_time] - pwv[upper_time - 1]) / (t_pwv[upper_time] - t_pwv[upper_time - 1])
            new_pwv = pwv[upper_time - 1] + a * (t_amps[i] - t_pwv[upper_time-1])
            out.append(new_pwv**2 * amps[i]**1.5)
        
    return out

# =============================================================================
# Function to compare data obtained from two different sources
# I.e., blood pressure from finapres and e-tattoo   
# =============================================================================
def compare_data(t_fina, dat_fina, t_ppg, dat_ppg, fina_threshold, fina_offset = 0):
    print("Comparing Finapres and PPG data...")
    aligned_dat_ppg = []
    aligned_dat_fina = []
    aligned_t = []
    
    for i in range (0, len(t_fina)):
        if(dat_fina[i] > fina_threshold and i > 10):
            aligned_dat_fina.append(dat_fina[i])
            aligned_t.append(t_fina[i])
            upper_time = bisect.bisect_left(t_ppg, t_fina[i])
            
            if upper_time >= len(t_ppg):
                upper_time = upper_time - len(t_ppg)
            
            #Check which PPG BP is closer in time
            if t_ppg[upper_time] - t_fina[i] > t_fina[i] - t_ppg[upper_time-1]:
                aligned_dat_ppg.append(dat_ppg[upper_time - 1])
            else:
                aligned_dat_ppg.append(dat_ppg[upper_time])

    print(len(aligned_dat_fina))
    print(len(aligned_dat_ppg))
    print(len(aligned_t))
    
    if fina_offset > 0:
        aligned_dat_fina = aligned_dat_fina[fina_offset:-1]
        aligned_dat_ppg = aligned_dat_ppg[0:len(aligned_dat_ppg)-fina_offset-1]
        aligned_t = aligned_t[0:len(aligned_t)-fina_offset-1]
    elif fina_offset < 0:
        aligned_dat_fina = aligned_dat_fina[0:len(aligned_dat_fina)+fina_offset-1]
        aligned_dat_ppg = aligned_dat_ppg[-fina_offset:-1]
        aligned_t = aligned_t[0:len(aligned_t)+fina_offset-1]    
    
    r = stats.pearsonr(aligned_dat_fina, aligned_dat_ppg)      
    print("Pearson's r: " + str(r))
    m, b = np.polyfit(aligned_dat_fina, aligned_dat_ppg, 1)
    print("Line of best fit: y = " + str(m) + "x + " + str(b))

    aligned_dat_ppg = aligned_dat_ppg / m - b/m
            
    fig = plt.figure(figsize=(12.8,4.8))
    ax1 = fig.add_axes([0.2, 0.2, 0.3, 0.7])
    ax2 = fig.add_axes([0.6, 0.2, 0.3, 0.7], sharey = ax1)
    
    ax1.plot(aligned_dat_fina, aligned_dat_ppg, "ko")
    #plt.plot(aligned_dat_fina, m*np.array(aligned_dat_fina) + b)
    ax1.plot(aligned_dat_fina, aligned_dat_fina)
    ax1.set_xlabel("Finapres BP (mm Hg)")
    ax1.set_ylabel("E-Tattoo (mm Hg)")
    
    ax2.scatter(aligned_t, aligned_dat_fina, label = "Finapres")
    ax2.scatter(aligned_t, aligned_dat_ppg, label = "E-Tattoo")
    ax2.legend()
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Pressure (mm Hg)")
    plt.show()      

# =============================================================================
# Extract the systolic upstroke time from a list of min and peak locations
# @param peaks locations of peaks
# @param min locations of mins
# @return List of sytolic upstroke times.
# =============================================================================
def sut(peaks, mins, fs):
    upstroke_time = [(p - m) / fs for p,m in zip(peaks, mins)]
    return upstroke_time


# =============================================================================
# Extracts the beat-to-beat heart rate from a list of peaks
# =============================================================================
def heart_rate(peaks, fs):
    heart_rate=[]
    for i in range(1,len(peaks)):
        heart_rate.append(1/((peaks[i]-peaks[i-1])/fs))
    return heart_rate