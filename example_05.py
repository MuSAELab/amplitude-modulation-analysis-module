#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Example 05
 This example shows the use of the transfomrs and their inverses 

 rfft()                            Fourier transform for real-valued signals 
 irfft()                           Inverse Fourier transform for real-valued signals 

 rfft_psd()                        Computes PSD data from x(f)
 irfft_psd()                       Recovers x(t) from its PSD data

 strfft_spectrogram()              Computes Spectrogram data using STFFT
 istrfft_modulation_spectrogram()  Recovers x(t) from its STFFT Spectrogram data 

 wavelet_spectrogram()             Computes Spectrogram using CWT
 iwavelet_modulation_spectrogram() Recovers x(t) from its CWT Spectrogram data
"""
import numpy as np
import matplotlib.pyplot as plt
from am_analysis import am_analysis as ama

if __name__ == "__main__":
    
    #Test signal  
    fs = 2000
    time_v = np.arange(10*fs)/fs
    
    f1 = np.sin(2 * np.pi * 100 * time_v)
    f2 = np.sin(2 * np.pi * 325 * time_v)
    m1 = np.sin(2 * np.pi * 5 * time_v) + 2
    m2 = np.sin(2 * np.pi * 3 * time_v) + 2
    
    xi = f1*m1 + f2*m2
    xi = xi[:, np.newaxis]
    n = xi.shape[0]
    
    plt.figure()
    ama.plot_signal(xi, fs)
    
    #%% time <--> frequency 
    #%% FFT and IFFT of a real-valued signal
    xi_rfft = ama.rfft(xi)
    xo = ama.irfft(xi_rfft, n)
    
    plt.figure()
    fi = plt.subplot(2,1,1)
    ama.plot_signal(xi, fs, 'Original x(t)')
    fo = plt.subplot(2,1,2, sharex=fi, sharey=fi)
    ama.plot_signal(xo, fs, 'Recovered x(t)')
    r = np.corrcoef(np.squeeze(xi), np.squeeze(xo))
    print('Correlation: ' + str(r[0,1]) + '\r\n' )

    #%% PSD data obtained with rFFT, and its inverse
    xi_psd = ama.rfft_psd(xi, fs)
    xo = ama.irfft_psd(xi_psd)
    
    plt.figure()
    fi = plt.subplot(4,1,1)
    ama.plot_signal(xi, fs, 'Original x(t)')
    fo = plt.subplot(4,1,4, sharex=fi, sharey=fi)
    ama.plot_signal(xo, fs, 'Recovered x(t)')
    plt.subplot(4,1,(2,3))
    ama.plot_psd_data(xi_psd)
    plt.title('PSD of x(t)')
    r = np.corrcoef(np.squeeze(xi), np.squeeze(xo))
    print('Correlation: ' + str(r[0,1]) + '\r\n' )

    #%% time <--> time-frequency 
    #%% STFT Spectrogram
    xi_strfft = ama.strfft_spectrogram(xi, fs, round(fs * 0.1), round(fs * 0.05))
    xo = ama.istrfft_spectrogram(xi_strfft)[0]
    
    plt.figure()
    fi = plt.subplot(4,1,1)
    ama.plot_signal(xi, fs, 'Original x(t)')
    fo = plt.subplot(4,1,4, sharex=fi, sharey=fi)
    ama.plot_signal(xo, fs, 'Recovered x(t)')
    plt.subplot(4,1,(2,3))
    ama.plot_spectrogram_data(xi_strfft)
    plt.title('STFT Spectrogram of x(t)')
    r = np.corrcoef(np.squeeze(xi), np.squeeze(xo))
    print('Correlation: ' + str(r[0,1]) + '\r\n' )

    #%% CWT Spectrogram
    xi_cwt = ama.wavelet_spectrogram(xi, fs)
    xo = ama.iwavelet_spectrogram(xi_cwt)
    
    plt.figure()
    fi = plt.subplot(4,1,1)
    ama.plot_signal(xi, fs, 'Original x(t)')
    fo = plt.subplot(4,1,4, sharex=fi, sharey=fi)
    ama.plot_signal(xo, fs, 'Recovered x(t)')
    plt.subplot(4,1,(2,3))
    ama.plot_spectrogram_data(xi_cwt)
    plt.title('CWT Spectrogram of x(t)')
    r = np.corrcoef(np.squeeze(xi), np.squeeze(xo))
    print('Correlation: ' + str(r[0,1]) + '\r\n' )

    #%% time <--> frequency-modulation-frequency 
    #%% STFT Modulation Spectrogram
    xi_mod_strfft = ama.strfft_modulation_spectrogram(xi, fs, round(fs * 0.1), round(fs * 0.05))
    xo = ama.istrfft_modulation_spectrogram(xi_mod_strfft)
    
    plt.figure()
    fi = plt.subplot(4,1,1)
    ama.plot_signal(xi, fs, 'Original x(t)')
    fo = plt.subplot(4,1,4, sharex=fi, sharey=fi)
    ama.plot_signal(xo, fs, 'Recovered x(t)')
    plt.subplot(4,1,(2,3))
    ama.plot_modulation_spectrogram_data(xi_mod_strfft, f_range=np.array([0, 1000]), modf_range=np.array([0, 10]))
    plt.title('STFT Modulation Spectrogram of x(t)')
    r = np.corrcoef(np.squeeze(xi), np.squeeze(xo))
    print('Correlation: ' + str(r[0,1]) + '\r\n' )
    
    #%% CWT Modulation Spectrogram
    xi_mod_cwt = ama.wavelet_modulation_spectrogram(xi, fs)
    xo = ama.iwavelet_modulation_spectrogram(xi_mod_cwt)
    
    plt.figure()
    fi = plt.subplot(4,1,1)
    ama.plot_signal(xi, fs, 'Original x(t)')
    fo = plt.subplot(4,1,4, sharex=fi, sharey=fi)
    ama.plot_signal(xo, fs, 'Recovered x(t)')
    plt.subplot(4,1,(2,3))
    ama.plot_modulation_spectrogram_data(xi_mod_cwt, f_range=np.array([0, 1000]), modf_range=np.array([0, 10]))
    plt.title('CWT Modulation Spectrogram of x(t)')
    r = np.corrcoef(np.squeeze(xi), np.squeeze(xo))
    print('Correlation: ' + str(r[0,1]) + '\r\n' )  











