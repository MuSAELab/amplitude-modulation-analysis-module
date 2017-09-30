#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function explore_stfft_ama_gui(X, fs, Name, c_map)
 Analysis of a Signal in Frequency-Frequency Domain
 Time -> Time-Frequency transformation performed with STFFT

 INPUTS:
  X     Real-valued column-vector signal or set of signals [n_samples, n_channels]
  fs    Sampling frequency (Hz)
 Optional:
  Name  (Optional) Name of the signal(s), List of Strings

"""

"""
Show how to connect to keypress events
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import ama_toolbox as ama

def press(event):
    global ix_segment
    global ix_channel
    global fig
    global cid
    
    #print('press', event.key)
    sys.stdout.flush()
    if event.key == 'left':       #Left arrow: Previous Segment   
        ix_segment = ix_segment - 1
        update_plots()
    elif event.key == 'right':    #Right arrow: Next Segment  
        ix_segment = ix_segment + 1
        update_plots()
    elif event.key == 'up':       #Up arrow: Previous Channel 
        ix_channel = ix_channel - 1
        first_run()
    elif event.key == 'down':     #Down arrow: Next Channel 
        ix_channel = ix_channel + 1
        first_run()
    elif event.key == 'a':        # A: Back 5 Segments    
        ix_segment = ix_segment - 1
        update_plots()
    elif event.key == 'd':        # D: Advance 5 Segments  
        ix_segment = ix_segment + 1
        update_plots()
    elif event.key == 'w':        # W: Previous 5 Channels 
        ix_channel = ix_channel - 1
        first_run()
    elif event.key == 's':        # S: Advance 5 Channels 
        ix_channel = ix_channel + 1
        first_run()   
    elif event.key == 'u':        # U: Update parameters
        update_parameters()
    elif event.key == 'escape':
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)
       
    return
        
def first_run():
    global ix_segment
    global ix_channel
    global n_segments
    global x_segments
    global win_size_smp    
    global win_shft_smp
    global x_probe
    global x_spectrogram
    global seg_size_sec
    global name


    ix_segment = 0
    print('Computing full-signal spectrogram...')

    # constrain ix_channel to [1 : n_channels]
    ix_channel = np.maximum(0, ix_channel)
    ix_channel = np.minimum(n_channels-1, ix_channel)

    # STFFT modulation spectrogram parameters in samples
    win_size_smp = round(win_size_sec * fs)  # (samples)
    win_shft_smp = round(win_shft_sec * fs)  # (samples)
    seg_size_smp = round(seg_size_sec * fs)  # (samples)
    seg_shft_smp = round(seg_shft_sec * fs)  # (samples)

    # signal for analysis
    x_probe = X[:, ix_channel]
    name = channel_names[ix_channel]

    # segment of signal under analysis
    x_segments, _, _= ama.epoching(x_probe, seg_size_smp, seg_size_smp - seg_shft_smp)
    n_segments = x_segments.shape[2]

    # compute and plot complete spectrogram
    x_spectrogram = ama.strfft_spectrogram(x_probe, fs, win_size_smp, win_shft_smp, channel_names = name);

    update_plots()
    return   

    
def update_plots():
    global ix_segment
    global x_segments
    global win_size_smp    
    global win_shft_smp
    global ix_channel
    global x_probe
    global x_spectrogram
    global fig
    global name
    
    fig.clear()
    
    # constrain ix_segment to [1 : n_segments]
    ix_segment = max([0, ix_segment])
    ix_segment = min([n_segments, ix_segment])

    # select segment
    x = x_segments[:, :, ix_segment]

    # compute and plot Modulation Spectrogram
    print('Computing modulation spectrogram...')
    x_stft_modspec = ama.strfft_modulation_spectrogram(x, fs, win_size_smp, win_shft_smp, fft_factor_y=2, fft_factor_x=2, channel_names=name)
    plt.subplot(4,2,(6,8))
    ama.plot_modulation_spectrogram_data(x_stft_modspec)
    #plot_modulation_spectrogram_data(x_stft_modspec, [], freq_range, mfreq_range, mfreq_color)

    # plot spectrogram for segment
    plt.subplot(4,2,7)
    ama.plot_spectrogram_data(x_stft_modspec['spectrogram_data'])
    #plot_spectrogram_data(x_stft_modspec.spectrogram_data, [], [], freq_range, freq_color)


    # plot time series for segment
    plt.subplot(4,2,5)
    ama.plot_signal(x, fs, name)
    plt.colorbar()
   
    # plot spectrogram for full signal        
    h_tf = plt.subplot(4,2,(3,4))
    ama.plot_spectrogram_data(x_spectrogram)
    #ama.plot_spectrogram_data(x_spectrogram, [], [], freq_range, freq_color)
    #set(gca, 'XLim', time_lim);

    # plot full signal
    h_ts = plt.subplot(4,2,(1,2))
    ama.plot_signal(x_probe, fs, name)
    plt.colorbar()

    # highlight area under analysis in time series
    seg_ini_sec = (ix_segment ) * seg_shft_sec
    seg_end_sec = seg_ini_sec + seg_size_sec

    plt.subplot(h_ts)
    varea([seg_ini_sec, seg_end_sec ],'r',0.4)
    
    # highlight area under analysis in Spectrogram
    plt.subplot(h_tf)
    varea([seg_ini_sec, seg_end_sec ],'r',0.4)


    print('done!')

    # display information about analysis
    print('signal name            : %s' % name )
    print('segment size  (seconds): %0.3f' % seg_size_sec)
    print('segment shift (seconds): %0.3f' % seg_shft_sec)
    print('segment position  (sec): %0.3f' % seg_ini_sec)
    print('window size   (seconds): %0.3f' % win_size_sec)
    print('window shift  (seconds): %0.3f' % win_shft_sec)
    print('windows per segment    : %d'% x_stft_modspec['n_windows'])


    fig.canvas.draw()
    return

def update_parameters():
    #global     
#TODO Obtain parameters from a GUI and update their global variables     
    #first_run()
    
    return


def varea(xlims, color_str, alpha_v=0.2):
    ax = plt.gca()
    ylims = ax.get_ylim()
    plt.fill((xlims[0], xlims[0], xlims[1], xlims[1]), 
             (ylims[0], ylims[1], ylims[1], ylims[0]), 
              color_str, alpha=alpha_v) 
    return



def explore_stfft_ama_gui(x, fs, channel_names = None, c_map = 'viridis'):
    
    # Global variables
    global ix_channel
    global ix_segment
    global n_channels
    global n_segments
    global x_segments
    global cid
    global fig
    
    global win_size_sec
    global win_shft_sec
    global seg_size_sec
    global seg_shft_sec
    
    global freq_range
    global mfreq_range
    global freq_color
    global mfreq_color
    
    global win_size_smp    
    global win_shft_smp
    
    global X
    global name
    global fs
    global channel_names
    
    # input 'x' as 2D matrix [samples, columns]
    try:
        x.shape[1]
    except IndexError:
        X = x[:, np.newaxis]
            
    # generate default channel names, if needed
    if channel_names is None:
        channel_names = []
        for ic  in range (0 , n_channels):
            icp = ic + 1
            channel_names.append( str('Signal-%02d' % icp) )
        
    
    # number of samples and number of channels
    n_channels = x.shape[1]
    
    #% Amplitude Modulation Analysis
    # Default Modulation Analysis parameters
    win_size_sec = 0.5      # window length for the STFFT
    win_shft_sec = 0.02     # shift between consecutive windows (seconds)
    seg_size_sec = 8.0      # segment of signal to compute the Modulation Spectrogram (seconds)
    seg_shft_sec = 8.0      # shift between consecutive segments (seconds)
    freq_range   = None     # limits [min, max] for the conventional frequency axis (Hz)
    mfreq_range  = None     # limits [min, max] for the modulation frequency axis (Hz)
    freq_color   = None     # limits [min, max] for the power in Spectrogram (dB)
    mfreq_color  = None     # limits [min, max] for the power in Modulation Spectrogram (dB)
        
    # initial channel and segment
    ix_channel = 0
    ix_segment = 0
    
    # other variables
    n_segments = None


    x_segments = None
    name = None
    win_size_smp = None
    win_shft_smp = None
    
    # Live GUI
    fig = plt.figure()
    first_run()
    cid = fig.canvas.mpl_connect('key_press_event', press)
     