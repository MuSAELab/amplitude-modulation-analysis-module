#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:45:19 2019

@author: cassani
"""

from . import am_analysis as ama
import numpy as np

def msqi_ama(x, fs):
    """
    Computes the Modulation Spectrum-Based ECG Quality Index (MSQI) for one or 
    many ECG signals defined in x, sampled with a sampling frequency fs

    Parameters
    ----------
    x  : 1D array with shape (n_samples) or
         2D array with shape (n_samples, n_signals)
    fs : Sampling frequency in Hz

    Returns
    -------
    msqi_value : MSQI value or values 
    hr_value   : HR values or values
    modulation_spectrogram : Structure or structures of modulation spectrogram
    
    See
    --------
    MS-QI: A Modulation Spectrum-Based ECG Quality Index for Telehealth Applications
    http://ieeexplore.ieee.org/document/6892964/
    
    D. P. Tobon V., T. H. Falk, and M. Maier, "MS-QI:  A  Modulation
    Spectrum-Based ECG Quality Index for Telehealth Applications", IEEE
    Transactions on Biomedical Engineering, vol. 63, no. 8, pp. 1613-1622,
    Aug. 2016    
    """
    
    # test ecg shape
    try:
        x.shape[1]
    except IndexError:
        x = x[:, np.newaxis]
    
    # Empirical values for the STFFT transformation
    win_size_sec  = 0.125   #seconds
    win_over_sec  = 0.09375 #seconds
    nfft_factor_1 = 16
    nfft_factor_2 = 4

    win_size_smp = int(win_size_sec * fs) #samples
    win_over_smp = int(win_over_sec * fs) #samples
    win_shft_smp = win_size_smp - win_over_smp

    # Computes Modulation Spectrogram
    modulation_spectrogram = ama.strfft_modulation_spectrogram(x, fs, win_size_smp, 
            win_shft_smp, nfft_factor_1, 'cosine', nfft_factor_2, 'cosine' )
    
    # Find fundamental frequency (HR)
    # f = (0, 40)Hz
    ix_f_00 = (np.abs(modulation_spectrogram['freq_axis'] -  0)).argmin(0)  
    ix_f_40 = (np.abs(modulation_spectrogram['freq_axis'] - 40)).argmin(0) + 1
    
    # Look for the maximum only from 0.6 to 3 Hz (36 to 180 bpm)
    valid_f_ix = np.logical_or(modulation_spectrogram['freq_mod_axis'] < 0.66 , modulation_spectrogram['freq_mod_axis'] > 3)
    
    # number of epochs
    n_epochs = modulation_spectrogram['power_modulation_spectrogram'].shape[2]
    
    msqi_vals = np.zeros(n_epochs)
    hr_vals   = np.zeros(n_epochs)
    
    for ix_epoch in range(n_epochs):
        B = np.sqrt(modulation_spectrogram['power_modulation_spectrogram'][:, :, ix_epoch])
    
        # Scale to maximun of B
        B = B / np.max(B)
    
        # Add B in the conventional frequency axis from 0 to 40 Hz
        tmp = np.sum(B[ix_f_00:ix_f_40, :], axis=0)
    
        # Look for the maximum only from 0.6 to 3 Hz (36 to 180 bpm)
        tmp[valid_f_ix] = 0
        ix_max = np.argmax(tmp)     
        freq_funda = modulation_spectrogram['freq_mod_axis'][ix_max]        
                   
        # TME
        tme = np.sum(B)
    
        eme = 0
        for ix_harm in range(1, 5):
            ix_fm = (np.abs(modulation_spectrogram['freq_mod_axis'] - (ix_harm * freq_funda) )).argmin(0) 
            ix_b = int(round(.3125 / modulation_spectrogram['freq_mod_delta'] ))  # 0.3125Hz, half lobe
            # EME
            eme = eme + np.sum(B[ 0 : ix_f_40, ix_fm - ix_b : ix_fm + ix_b + 1 ])  
        
        # RME
        rme = tme - eme
        # MS-QI
        msqi_vals[ix_epoch] = eme / rme
        # HR
        hr_vals[ix_epoch] = freq_funda * 60
         
    return (msqi_vals, hr_vals, modulation_spectrogram)