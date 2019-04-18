#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Example 04
This example shows the amplitude modulation analysis toolbox for speech data

 rfft_psd()                        Compute PSD using rFFT
 strfft_spectrogram()              Compute Spectrogram using STFFT
 strfft_modulation_spectrogram()   Compute Modulation Spectrogram using STFFT
 wavelet_spectrogram()             Compute Spectrogram using wavelet transformation
 wavelet_modulation_spectrogram()  Compute Modulation Spectrogram using wavelet transformation

 plot_signal()                      Plot a signal in time domain
 plot_psd_data()                    Plot PSD data obtained with rfft_psd()
 plot_spectrogram_data()            Plot Spectrogram data obtained with
                                         strfft_spectrogram() or wavelet_spectrogram()
 plot_modulation_spectrogram_data() Plot Modulation Spectrogram data obtained with
                                         strfft_modspectrogram() or wavelet_modspectrogram()

 Moreover, this script compares diverse ways to compute the power of a
 signal in the Time, Frequency, Time-Frequency and Frequency-Frequency domains

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from am_analysis import am_analysis as ama

if __name__ == "__main__":
    
    # speech signal
    # The speech signal p234_004.wav is one sample from the: 
    # CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit
    # avialable in: https://datashare.is.ed.ac.uk/handle/10283/2651
    fs, x = wavfile.read('./example_data/p234_004.wav')
    x_name = ['speech']
    x = x / 32768
    # 1s segment to analyze
    x = x[int(fs*1.6) : int(fs*3.6)]

    
    #%% STFT-based
    # Parameters
    win_size_sec = 0.04  # window length for the STFFT (seconds)
    win_shft_sec = 0.01  # shift between consecutive windows (seconds)

    plt.figure()
        
    stft_spectrogram = ama.strfft_spectrogram(x, fs, win_size = round(win_size_sec*fs), win_shift = round(win_shft_sec*fs), channel_names = x_name)
    plt.subplot2grid((4,5),(1,0),rowspan=1, colspan=5)
    ama.plot_spectrogram_data(stft_spectrogram)

    plt.subplot2grid((4,5),(0,0),rowspan=1, colspan=5)
    ama.plot_signal(x, fs, x_name) 
    plt.colorbar()
    
    stft_modulation_spectrogram = ama.strfft_modulation_spectrogram(x, fs, win_size = round(win_size_sec*fs), win_shift = round(win_shft_sec*fs), channel_names = x_name)
    plt.subplot2grid((4,5), (2,1),rowspan=2, colspan=3)  
    ama.plot_modulation_spectrogram_data(stft_modulation_spectrogram, 0 , modf_range = np.array([0,20]), c_range =  np.array([-90, -50]))
    
    
    #%% Parameters for CWT for speech signal 
    n_cycles = 6          # number of cycles (for Complex Morlet)
    up_lim = np.floor(np.log(fs/2) / np.log(2))
    frequency_vector = 2**np.arange(1,up_lim,0.2) # vector of frequencies to compute the CWT
    
    plt.figure()
    
    cwt_spectrogram = ama.wavelet_spectrogram(x, fs, n_cycles, frequency_vector, channel_names=x_name)
    plt.subplot2grid((4,5),(1,0),rowspan=1, colspan=5)
    ama.plot_spectrogram_data(cwt_spectrogram)
    
    plt.subplot2grid((4,5),(0,0),rowspan=1, colspan=5)
    ama.plot_signal(x, fs, x_name) 
    plt.colorbar()
    
    cwt_modulation_spectrogram = ama.wavelet_modulation_spectrogram(x, fs, n_cycles, frequency_vector, channel_names = x_name)
    plt.subplot2grid((4,5), (2,1),rowspan=2, colspan=3)  
    ama.plot_modulation_spectrogram_data(cwt_modulation_spectrogram, 0, modf_range=np.array([0,20]), c_range=np.array([-90, -50]))
    
    
        
