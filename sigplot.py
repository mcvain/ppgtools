import matplotlib.pyplot as plt
import numpy as np

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']


#This package is used for plotting multiple BioSignals.    
def plot_biosignals(signals, event_markers = [], units = [], peaks = []):
    '''
    Function to plot biosignals.

    Parameters
    ----------
    signals : array_like of BioSignal
        Array of BioSignals to plot.
    event_markers : array_like of EventMarker, optional
        Add if event markers should be plotted. The default is [].
    units : array_like, optional
        Array of strings. This replaces the y-axis title. The default is [].
    peaks : array_like, optional
        Array of arrays. Each inner array is the locations of peaks to be plotted. The function will automatically plot the correct amplitude. 
        The outer array length must be the same length as the number of signals.
        If a signal does not need peaks plotted, input an empty inner array.
        The default is [].

    Returns
    -------
    None.

    '''
    if len(units) != len(signals):
        units = ["Amplitude [a.u.]" for i in range(len(signals))]
    
    
    num_ax = len(signals)
    if len(event_markers) > 0:
        num_ax +=1
    
    fig, axs = plt.subplots(num_ax, 1, sharex = True)
    
    for i in range (0, len(signals)):
        t = np.linspace(0, len(signals[i].data) / signals[i].fs, len(signals[i].data))
        
        ax = axs[i]
        
        ax.set_ylabel(units[i])        
        ax.plot(t, signals[i].data, c = colors[i % len(colors)])
        ax.set_title(signals[i].name,  position=(0.025, 0.05), horizontalalignment='left')
            
        for j in range (0, len(event_markers)):
            ax.axvline(event_markers[j].t, c = 'red')
            
        #If there are peaks to plot
        if len(peaks) == len(signals):
            ax.plot(t[peaks[i]], signals[i].data[peaks[i]], 'ro')
      
        
    #Plot the event markers
    last_xpoint = -1
    y_point = 0
    for i in range (0, len(event_markers)):
        if last_xpoint == event_markers[i].t:
            y_point -= 100
        else:
            y_point = 0
            last_xpoint = event_markers[i].t
        axs[len(signals)].annotate(event_markers[i].label, [event_markers[i].t, y_point], rotation = 'vertical', size = 6)
        axs[len(signals)].axvline(event_markers[i].t, c = 'red')
        
        
        axs[len(signals)].set_yticklabels([])
        axs[len(signals)].set_yticks([])
        axs[len(signals)].spines['top'].set_visible(False)
        axs[len(signals)].spines['right'].set_visible(False)
        axs[len(signals)].spines['bottom'].set_visible(False)
        axs[len(signals)].spines['left'].set_visible(False)
        
        
        #axs[len(signals)].axis('off')
        

    fig.legend()
    plt.xlabel("Time [s]")
    plt.xlim([550, 1600])
    