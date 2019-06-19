#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%% Example MSQI
% This example shows the use of the amplitude modulation analysis toolkit
% to compute the Modulation Spectrum-Based ECG Quality Index (MSQI) as a 
% blind metric to measure the signal-to-noise (SNR) ration in ECG signals
% 
% The MSQI was originally presented in:
%
% D. P. Tobon V., T. H. Falk, and M. Maier, "MS-QI:  A  Modulation
% Spectrum-Based ECG Quality Index for Telehealth Applications", IEEE
% Transactions on Biomedical Engineering, vol. 63, no. 8, pp. 1613-1622,
% Aug. 2016

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from am_analysis import am_analysis as ama
from am_analysis.msqi_ama import msqi_ama


#%% ECG signal 
# load ECG signal
# x_clean is a 5-s segment of clean ECG signal
# x_noisy is the x_clean signal contaminated with pink noise to have a 0db SNR
[x_clean, x_noisy, fs] = pickle.load(open( "./example_data/msqi_ecg_data.pkl", "rb" ))

#%% Compute MSQI and heart rate (HR), and plot modulation spectrogram
[msqi_clean, hr_clean, modulation_spectrogram_clean] = msqi_ama(x_clean, fs)
[msqi_noisy, hr_noisy, modulation_spectrogram_noisy] = msqi_ama(x_noisy, fs)

print('HR = {:0.2f} bpm'.format(hr_clean[0]))
print('MSQI for clean ECG = {:0.3f}'.format(msqi_clean[0]))
print('MSQI for noisy ECG = {:0.3f}'.format(msqi_noisy[0]))

#%% Plot modulation spectrograms      
plt.figure()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
# clean 
plt.subplot(3,2,1)
ama.plot_signal(x_clean, fs)
plt.title('Clean ECG')
plt.subplot(3,2,(3,5))
ama.plot_modulation_spectrogram_data(modulation_spectrogram_clean, f_range=np.array([0, 60]), c_range=np.array([-90, -40]))
plt.title('Modulation spectrogram clean ECG')
plt.subplot(3,2,2)
ama.plot_signal(x_noisy, fs)
plt.title('Noisy ECG')
plt.subplot(3,2,(4,6))
ama.plot_modulation_spectrogram_data(modulation_spectrogram_noisy, f_range=np.array([0, 60]), c_range=np.array([-90, -40]))
plt.title('Modulation spectrogram noisy ECG')
