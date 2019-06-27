#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Example 02
 Script to show the use of the functions:

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
from am_analysis import am_analysis as ama

if __name__ == "__main__":
    
    #Test signal
    
    fs = 500
    T = 1/fs
    t1_vct = np.arange(10*fs)/fs
    
    x1 = 3 * np.sin (2 * np.pi * 10 * t1_vct)
    x2 = 2 * np.sin (2 * np.pi * 24 * t1_vct)
    x3 = 1 * np.random.randn(t1_vct.shape[0])
    
    x = np.concatenate([x1, x2, x3])
    
    n = x.shape[0]
    
    # Plot signal
    plt.figure()
    ama.plot_signal(x, fs, 'test-signal')
    
    # Power in Time Domain
    # Energy of the signal
    energy_x = T * sum(x**2)
    duration = T * n
    
    # Power of the signal
    power_x = energy_x / duration
    
    # A simpler way is
    power_x_2 = (1 / n) * sum(x**2)
    
    # Power in Frequency domain
    # Power using FFT
    X = np.fft.fft(x)
    power_X = float( (1 / n**2) * sum(np.abs(X)**2))
    
    # Power using its PSD from rFFT
    psd_rfft_r = ama.rfft_psd(x, fs, win_function = 'boxcar')
    f_step = psd_rfft_r['freq_delta']
    power_psd_rfft_x_rw = f_step * sum(psd_rfft_r['PSD'])[0]
    plt.figure()
    ama.plot_psd_data(psd_rfft_r)
    
    # Power using its PSD from rFFT
    psd_rfft_b = ama.rfft_psd(x, fs, win_function = 'blackmanharris')
    f_step = psd_rfft_r['freq_delta']
    power_psd_rfft_x_bh = f_step * sum(psd_rfft_r['PSD'])[0]
    plt.figure()
    ama.plot_psd_data(psd_rfft_b)
    
    # Power from STFFT Spectrogram (Hamming window)
    w_size =  1 * fs
    w_shift = 0.5 * w_size
    rfft_spect_h = ama.strfft_spectrogram(x, fs, w_size, w_shift, win_function = 'hamming' )
    power_spect_h = sum(sum(rfft_spect_h['power_spectrogram']))[0] * rfft_spect_h['freq_delta'] * rfft_spect_h['time_delta']
    plt.figure()
    ama.plot_spectrogram_data(rfft_spect_h)
    
    # Power from STFFT Spectrogram (Rectangular window)
    w_size =  1 * fs
    w_shift = 0.5 * w_size
    rfft_spect_r = ama.strfft_spectrogram(x, fs, w_size, w_shift, win_function = 'boxcar')
    power_spect_r = sum(sum(rfft_spect_r['power_spectrogram']))[0] * rfft_spect_r['freq_delta'] * rfft_spect_r['time_delta']
    plt.figure()
    ama.plot_spectrogram_data(rfft_spect_r)
    
    # Power from Wavelet Spectrogram N = 6
    wav_spect_6 = ama.wavelet_spectrogram(x, fs, 6)
    power_wav_6 = sum(sum(wav_spect_6['power_spectrogram']))[0] * wav_spect_6['freq_delta'] * wav_spect_6['time_delta'] 
    plt.figure()
    ama.plot_spectrogram_data(wav_spect_6)
    
    # Power from Wavelet Spectrogram N = 10
    wav_spect_10 = ama.wavelet_spectrogram(x, fs, 10)
    power_wav_10 = sum(sum(wav_spect_10['power_spectrogram']))[0] * wav_spect_10['freq_delta'] * wav_spect_10['time_delta'] 
    plt.figure()
    ama.plot_spectrogram_data(wav_spect_10)
    
    # Power from Modulation Spectrogram STFFT
    w_size =  1 * fs
    w_shift = 0.5 * w_size
    rfft_mod_b = ama.strfft_modulation_spectrogram(x, fs, w_size, w_shift, win_function_y = 'boxcar', win_function_x = 'boxcar')
    power_mod = sum(sum(rfft_mod_b['power_modulation_spectrogram']))[0] * rfft_mod_b['freq_delta'] * rfft_mod_b['freq_mod_delta']
    plt.figure()
    ama.plot_modulation_spectrogram_data(rfft_mod_b)
    
    # Power from Modulation Spectrogram Wavelet
    wav_mod_6 = ama.wavelet_modulation_spectrogram(x, fs, 6, win_function_x = 'boxcar')
    power_mod_w = sum(sum(wav_mod_6['power_modulation_spectrogram']))[0] * wav_mod_6['freq_delta'] * wav_mod_6['freq_mod_delta']
    plt.figure()
    ama.plot_modulation_spectrogram_data(wav_mod_6)
    
    plt.show()

    
