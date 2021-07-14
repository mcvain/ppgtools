'''
BioSignal Class
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from ppgtools import sigpro, sigpeaks

class BioSignal:   
    def __init__(self, index, name, bytes_per_point, fs, bit_resolution, signed, little_endian):
        self.index = index
        self.name = name
        self.bytes_per_point = bytes_per_point
        self.fs = fs
        self.bit_resolution = bit_resolution
        self.signed = signed
        self.little_endian = little_endian
        self.data = np.zeros(0)
        
    def getEndian(self):
        if self.little_endian:
            return 'little'
        return 'big'

    def resample(self, tfs, aa = False, change_self = True):
        '''
        Resample the data to a new sampling rate using Fourier method.
        Recommend tfs being an integer factor of fs.

        Parameters
        ----------
        tfs : int
            Target sample rate.
        aa : bool, optional
            Set to true if an anti-aliasing filter should be used before resampling. The default is False.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = sigpro.resample(self.data, self.fs, tfs, aa)
        
        if change_self:
            self.data = output
            self.fs = tfs
        
        return output

    def downsample(self, tfs, aa = False, change_self = True):
        '''
        Downsamples the data to a new sampling rate by decimation.
        Recomend the target sampling rate to be an integer factor of the original.

        Parameters
        ----------
        tfs : int
            Target sample rate in Hz.
        aa : bool, optional
            Set to true if an anti-aliasing filter should be used before downsampling. The default is False.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        
        output = sigpro.downsample(self.data, self.fs, aa)
        
        if change_self:
            self.data = output
            self.fs = tfs
            
        return output
        
    
    def upsample(self, num_padding, change_self = True):
        '''
        Helper function that will upsample the data to a higher sampling rate.

        Parameters
        ----------
        num_padding : int
            How many zeros to put between each data point.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
    
        output = sigpro.upsample(self.data, self.fs, num_padding)
        
        if change_self:
            self.data = output
            self.fs = self.fs * (1 + num_padding)
            
        return output        
    

    def remove_mean(self, change_self = True):
        '''
        Removes the mean from the data i.e., make the DC offset 0.
        
        Parameters
        ----------
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = sigpro.remove_mean(self.data)
        
        if change_self:
            self.data = output
        
        return output
    
    def filter_signal(self, filter_pass, fc, order = 4, filter_design = "butter", zero_phase = True, change_self = True):  
        '''
        Filters the signal

        Parameters
        ----------
        filter_pass : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
            Filter type
        fc : array_like
            Critical freqencies.
        order : int, optional
            The order of the filter. The default is 4.
        filter_design : {'butter', 'bessel', 'chebyii', 'fir'}, optional
            How the filter coefficients are calculated. The default is "butter".
        zero_phase : bool, optional
            If true, uses filtfilt. Otherwise, uses lfilter. The default is True.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.
            
        Raises
        ------
        ValueError
            If a filter_design is not recognized, it will throw this error.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = sigpro.filter_signal(self.data, self.fs, filter_pass, fc, order, filter_design, zero_phase)
                    
        if change_self:
            self.data = output
        
        return output    

    def calc_fft(self, positive_only = False, plot = False):
        '''
        Calculates and returns the FFT of the data.

        Parameters
        ----------
        positive_only : bool, optional
            If true, only the positive frequencies of the FFT will be returned. The default is False.
        plt : bool, optional
            If true, the FTT will be plotted. The default is False.
        
        Returns
        -------
        freq : array_like
            The DST sample frequencies.
        signal_fx : array_like
            The FFT complex amplitude values.

        '''
        
        return sigpro.calc_fft(self.data, self.fs, positive_only, plot)
        

    def trim_signal(self, lower_bound = 0, upper_bound = 0, change_self = True):
        '''
        Truncates the signal.

        Parameters
        ----------
        lower_bound : float, optional
            Where to start the signal, in seconds. The default is 0.
        upper_bound : float, optional
            Where to cut off the signal (original data), in seconds. The default is 0.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.
        
        Returns
        -------
        output : array_like
            The data after the change.

        '''
        output = sigpro.trim_signal(self.data, self.fs, lower_bound, upper_bound)
        
        if change_self:
            self.data = output
        
        return output
        
    def plot(self):
        plt.figure()
        t = np.linspace(0, len(self.data) / self.fs, len(self.data))
        plt.plot(t, self.data)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [a.u.]')
        plt.yticks([])
        plt.title(self.name)
        
    def convert_Si7050(self):
        self.data = self.data * 175.72 / 65536 - 46.85

    def shift(self, delay_amount, change_self = True):
        '''
        Function to shift the signal in time.

        Parameters
        ----------
        delay_amount : float
            The amount to delay the signal, in seconds. A positive amount will move up the signal, and a negative amount will truncate the end of the signal.
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        
        output = sigpro.shift(self.data, self.fs, delay_amount)
        
        if change_self:
            self.data = output
            
        return output
    
    def invert(self, change_self = True):
        '''
        Function to flip the waveform

        Parameters
        ----------
        change_self : bool, optional
            This determines if the signal will actually alter itself, or just return without altering. The defalut is True.

        Returns
        -------
        output : array_like
            The data after the change.

        '''
        
        output = sigpro.invert(self.data)
        
        if change_self:
            self.data = output
            
        return output
    
    def find_peaks_adaptive_threshold(self, window_size, fc = []):
        '''
        Function to determine the location of peaks using an adaptive threshold.
        The adaptive threshold is obtained from the moving average of the original signal.
        
        NOTE: This takes the window size in seconds, as opposed to samples.
    
        Parameters
        ----------
        inp : array_like
            Signal to find peaks in.
        window_size : int
            Moving average size in seconds.
    
        Returns
        -------
        peaks : array_like
            Indices of the estimated peaks.
    
        '''
        
        peaks = sigpeaks.find_peaks_adaptive_threshold(self.data, int(window_size * self.fs))
        return peaks
    
    def find_peaks_pan_tompkins(self):
        peaks = sigpeaks.find_peaks_pan_tompkins(self.data, self.fs)
        
        return peaks