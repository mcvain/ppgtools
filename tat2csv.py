def tat2csv(tat, save_location=None):  
    '''
    Converts the e-tattoo data recording (.tat) into a CSV file 

    Parameters
    ----------
    tat : str
        Location of the recording. 
    save_location : str, optional, or None by default 
        Location to save the .csv output. If None, the file is saved at the current directory. 

    Returns
    -------
    raw_array : array_like
            The data after the conversion. Can be imported directly into MNE-Python, etc. 

    '''

    