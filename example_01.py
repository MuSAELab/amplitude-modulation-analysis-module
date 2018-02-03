#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Example 01
 This example shows the use of the GUI to explore Amplitude Modulations
 for ECG data and EEG data

 The 'explore_strfft_am_gui()' computes the Modulation Spectrogram.  
   It uses the Short Time Real Fourier Fast Transform (STRFFT) to compute 
   the Spectrogram, after rFFT is used to obtain the Modulation Spectrogram

 The 'explore_wavelet_am_gui()' computes the Modulation Spectrogram using 
   It uses the Wavelet transform with Complex Morlet wavelet to compute 
   the Spectrogram, after rFFT is used to obtain the Modulation Spectrogram

 Usage for explore_*_am_gui()

 Once the GUI is executed, it accepts the following commands

   Key          Action
   Up Arrow     Previous channel                  (-1 channel) 
   Down Arrow   Next channel                      (+1 channel)
   Left Arrow   Go back to the previous segment   (-1 segment shift)  
   Right Arrow  Advance to the next segment       (+1 segment shift)
   'W'          Previous channel                  (-5 channels) 
   'S'          Next channel                      (+5 channel)
   'A'          Go back to the previous segment   (-5 segment shift) 
   'D'          Advance to the next segment       (+5 segment shift)
 
   'U'          Menu to update:
                   parameters for Modulation Spectrogram 
                   ranges for conventional and modulation frequency axes
                   ranges for power in Spectrogram and Modulation Spectrogram 
   ESC          Close the GUI
"""

import pickle
from am_analysis.explore_stfft_ama_gui import explore_stfft_ama_gui
from am_analysis.explore_wavelet_ama_gui import explore_wavelet_ama_gui


if __name__ == "__main__":
    
    #% ECG data (1 channel) using STFFT-based Modulation Spectrogram
    [x, fs] = pickle.load(open( "./example_data/ecg_data.pkl", "rb" ))
    # STFFT Modulation Spectrogram
    explore_stfft_ama_gui(x, fs, ['ECG'])
    
    #% ECG data (1 channel) using wavelet-based Modulation Spectrogram
    [x, fs] = pickle.load(open( "./example_data/ecg_data.pkl", "rb" ))
    # Wavelet Modulation Spectrogram
    explore_wavelet_ama_gui(x, fs, ['ECG'])
    
    #% EEG data (7 channels) using STFFT-based Modulation Spectrogram
    [x, fs, ch_names] = pickle.load(open( "./example_data/eeg_data.pkl", "rb" ))
    # STFFT Modulation Spectrogram
    explore_stfft_ama_gui(x, fs, ch_names)
    
    #% EEG data (7 channels) using wavelet-based Modulation Spectrogram
    [x, fs, ch_names] = pickle.load(open( "./example_data/eeg_data.pkl", "rb" ))
    # Wavelet Modulation Spectrogram
    explore_wavelet_ama_gui(x, fs, ch_names)
    