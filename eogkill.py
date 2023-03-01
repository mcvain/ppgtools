import numpy as np
from covar import cov_shrink_ss, cov_shrink_rblw  # different methods for cov_shrink are available

class eyereg():
    def __init__(self):
        self.W = 0
        self.eeg_chan_idxs = 0
        self.eog_chan_idxs = 0

    def fit(self, data, labels, eeg_chan_idxs, eog_chan_idxs):
        # fits the regression coefficients for eye movement and blink periods
        # data: 3D array of shape (n_channels, n_samples, n_epochs), OR
        #       2D array of shape (n_channels, n_samples)
        # labels: corresponding sized array but with n_channels = 1. 
        #         label=1 ... rightwards eye movement
#                       2 ... leftwards eye movement
#                       3 ... upwards eye movement
#                       4 ... downwards eye movement
#                       5 ... blink
#                       6 ... resting activity
#                       0 ... none

        # grab data for fitting 
        b = labels < 6

        X_art = data[eog_chan_idxs, :][:, b.ravel()]
        Y_art = data[eeg_chan_idxs, :][:, b.ravel()]

        Z = np.concatenate((X_art, Y_art), axis=0)
        nc_x = np.size(X_art, 0)
        nc_y = np.size(Y_art, 0)

        Czz = cov_shrink_ss(Z.T)[0]  # python cov_shrink_ss gives the matrix as the first returned value

        Cxx = Czz[:nc_x, :nc_x]
        Cyx = Czz[nc_x:(nc_x+nc_y), :nc_x]

        # compute minimum mean squared error weight matrix W
        # self.W = Cyx / Cxx 
        self.W = np.matmul(Cyx, np.linalg.pinv(Cxx))  # verified on MATLAB to be the same thing as Cyx/Cxx
        self.eeg_chan_idxs = eeg_chan_idxs
        self.eog_chan_idxs = eog_chan_idxs

        print("EOG artifact correction matrix W: ", self.W)

        return

    def apply(self, data):
        # applies eog artifact correction to new data samples
        data[self.eeg_chan_idxs, :] = data[self.eeg_chan_idxs, :] - self.W @ data[self.eog_chan_idxs, :]

        return data
    
