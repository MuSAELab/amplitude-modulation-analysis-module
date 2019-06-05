#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function explore_stfft_ama_gui(X, fs, Names, c_map)
 Analysis of a Signal in Frequency-Frequency Domain
 Time -> Time-Frequency transformation performed with STFFT

 INPUTS:
  X     Real-valued column-vector signal or set of signals [n_samples, n_channels]
  fs    Sampling frequency (Hz)
 Optional:
  Names (Optional) Name of the signal(s), List of Strings
  c_map (Optional) Colormap, Default 'viridis'

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button


from . import am_analysis as ama

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
        fig.canvas.mpl_disconnect(cid)
        create_parameter_gui()
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
    global fs
    global X
    global parameters
    global channel_names

    ix_segment = 0
    print('Computing full-signal spectrogram...')

    # constrain ix_channel to [1 : n_channels]
    ix_channel = np.maximum(0, ix_channel)
    ix_channel = np.minimum(n_channels-1, ix_channel)

    # STFFT modulation spectrogram parameters in samples
    win_size_smp = round(parameters['win_size_sec'] * fs)  # (samples)
    win_shft_smp = round(parameters['win_shft_sec'] * fs)  # (samples)
    seg_size_smp = round(parameters['seg_size_sec'] * fs)  # (samples)
    seg_shft_smp = round(parameters['seg_shft_sec'] * fs)  # (samples)

    # signal for analysis
    x_probe = X[:, ix_channel]

    # segment of signal under analysis
    x_segments, _, _= ama.epoching(x_probe, seg_size_smp, seg_size_smp - seg_shft_smp)
    n_segments = x_segments.shape[2]

    # compute and plot complete spectrogram
    x_spectrogram = ama.strfft_spectrogram(x_probe, fs, win_size_smp, win_shft_smp, channel_names=[channel_names[ix_channel]])

    update_plots()
    return   

def create_parameter_gui():
    global fig2
    global boxes
    global parameters
    # new figure for parameters
    fig2, ax2 = plt.subplots()
    plt.axis('off')   
    plt.subplots_adjust(top=0.5, left=0.1, right=0.2, bottom=0.4)
    
    axbox = plt.axes([0.4, 0.85, 0.2, 0.075])
    text_box0 = TextBox(axbox, 'Segment (seconds)', str(parameters['seg_size_sec']))
    axbox = plt.axes([0.4, 0.75, 0.2, 0.075])
    text_box1 = TextBox(axbox, 'Segment shift (seconds)', str(parameters['seg_shft_sec']))
    axbox = plt.axes([0.4, 0.65, 0.2, 0.075])
    text_box2 = TextBox(axbox, 'Window size (seconds)', str(parameters['win_size_sec']))
    axbox = plt.axes([0.4, 0.55, 0.2, 0.075])
    text_box3 = TextBox(axbox, 'Window shift (seconds)', str(parameters['win_shft_sec']))
    axbox = plt.axes([0.4, 0.45, 0.2, 0.075])
    text_box4 = TextBox(axbox, 'Freq Conv. min Max (Hz)', str(parameters['freq_range']).strip('[').strip(']') )
    axbox = plt.axes([0.4, 0.35, 0.2, 0.075])
    text_box5 = TextBox(axbox, 'Spectr Pwr min Max (dB)', str(parameters['freq_color']).strip('[').strip(']'))
    axbox = plt.axes([0.4, 0.25, 0.2, 0.075])
    text_box6 = TextBox(axbox, 'Freq Mod. min Max (Hz)', str(parameters['mfreq_range']).strip('[').strip(']'))
    axbox = plt.axes([0.4, 0.15, 0.2, 0.075])
    text_box7 = TextBox(axbox, 'ModSpec Pwr min Max (dB)', str(parameters['mfreq_color']).strip('[').strip(']'))
    
    axbox = plt.axes([0.4, 0.05, 0.2, 0.075])
    ok_button = Button(axbox, 'OK')
    
    boxes = [text_box0, text_box1, text_box2, text_box3,
             text_box4, text_box5, text_box6, text_box7, ok_button]
    
    ok_button.on_clicked(submit)
    return

def submit(text):
    global fig2
    global fig
    global cid
    plt.close(fig2)
    update_parameters()
    cid = fig.canvas.mpl_connect('key_press_event', press)

def update_parameters():
    global boxes
    global parameters
    # Pop reference to Button
    boxes.pop(-1)
    parameters['seg_size_sec'] = float(boxes[0].text)
    parameters['seg_shft_sec'] = float(boxes[1].text)
    parameters['win_size_sec'] = float(boxes[2].text)
    parameters['win_shft_sec'] = float(boxes[3].text)
    if boxes[4].text == 'None':
        parameters['freq_range'] = None
    else:
        parameters['freq_range'] = np.fromstring(boxes[4].text, sep=' ')
    if boxes[5].text == 'None':
        parameters['freq_color'] = None
    else:
        parameters['freq_color'] = np.fromstring(boxes[5].text, sep=' ')
    if boxes[6].text == 'None':
        parameters['mfreq_range'] = None
    else:
        parameters['mfreq_range'] = np.fromstring(boxes[6].text, sep=' ')
    if boxes[7].text == 'None':
        parameters['mfreq_color'] = None
    else:
        parameters['mfreq_color'] = np.fromstring(boxes[7].text, sep=' ')

    first_run()
    
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
    global fs
    global parameters
    global gc_map
    global channel_names
    
    fig.clear()
    
    # constrain ix_segment to [1 : n_segments]
    ix_segment = max([0, ix_segment])
    ix_segment = min([n_segments, ix_segment])

    # select segment
    x = x_segments[:, :, ix_segment]

    # compute and plot Modulation Spectrogram
    print('Computing modulation spectrogram...')
    x_stft_modspec = ama.strfft_modulation_spectrogram(x, fs, win_size_smp, win_shft_smp, fft_factor_y=2, fft_factor_x=2, channel_names=[channel_names[ix_channel]])
    plt.subplot(4,2,(6,8))
    ama.plot_modulation_spectrogram_data(x_stft_modspec, f_range=parameters['freq_range'], modf_range=parameters['mfreq_range'], c_range=parameters['mfreq_color'], c_map=gc_map)

    # plot time series for segment
    plt.subplot(4,2,5)
    ama.plot_signal(x, fs, channel_names[ix_channel])
    plt.colorbar()
    time_lim = plt.xlim()
    
    # plot spectrogram for segment
    plt.subplot(4,2,7)
    ama.plot_spectrogram_data(x_stft_modspec['spectrogram_data'], f_range=parameters['freq_range'], c_range=parameters['freq_color'], c_map=gc_map )
    plt.xlim(time_lim)

    # plot full signal
    h_ts = plt.subplot(4,2,(1,2))
    ama.plot_signal(x_probe, fs, channel_names[ix_channel])
    plt.colorbar()
    time_lim = plt.xlim()
   
    # plot spectrogram for full signal        
    h_tf = plt.subplot(4,2,(3,4))
    ama.plot_spectrogram_data(x_spectrogram, f_range=parameters['freq_range'], c_range=parameters['freq_color'], c_map=gc_map )
    plt.xlim(time_lim)
    
    # highlight area under analysis in time series
    seg_ini_sec = (ix_segment ) * parameters['seg_shft_sec']
    seg_end_sec = seg_ini_sec + parameters['seg_size_sec']

    plt.subplot(h_ts)
    varea([seg_ini_sec, seg_end_sec ],'r',0.4)
    
    # highlight area under analysis in Spectrogram
    plt.subplot(h_tf)
    varea([seg_ini_sec, seg_end_sec ],'r',0.4)

    print('done!')

    # display information about analysis
    print('signal name            : %s' % channel_names[ix_channel] )
    print('segment size  (seconds): %0.3f' % parameters['seg_size_sec'])
    print('segment shift (seconds): %0.3f' % parameters['seg_shft_sec'])
    print('segment position  (sec): %0.3f' % seg_ini_sec)
    print('window size   (seconds): %0.3f' % parameters['win_size_sec'])
    print('window shift  (seconds): %0.3f' % parameters['win_shft_sec'])
    print('windows per segment    : %d'% x_stft_modspec['n_windows'])

    fig.canvas.draw()
    plt.show()
    return


def varea(xlims, color_str, alpha_v=0.2):
    ax = plt.gca()
    ylims = ax.get_ylim()
    plt.fill((xlims[0], xlims[0], xlims[1], xlims[1]), 
             (ylims[0], ylims[1], ylims[1], ylims[0]), 
              color_str, alpha=alpha_v) 
    return



def explore_stfft_ama_gui(x, fs_arg, channel_names_arg = None, c_map = 'viridis'):
    
    # Global variables
    global ix_channel
    global ix_segment
    global n_channels
    global n_segments
    global x_segments
    global cid
    global fig
    
    global parameters
    global gc_map
        
    global win_size_smp    
    global win_shft_smp
    
    global X
    global name
    global fs
    global channel_names
    
    fs = fs_arg
    channel_names = channel_names_arg
    X = x
    gc_map = c_map
    
    # input 'x' as 2D matrix [samples, columns]
    try:
        X.shape[1]
    except IndexError:
        X = X[:, np.newaxis]
    
    # number of channels
    n_channels = X.shape[1]
    
    if type(channel_names) == str and n_channels == 1:
        channel_names = [channel_names]
    # generate default channel names, if needed
    if channel_names is None or len(channel_names) != n_channels:
        channel_names = []
        for ic  in range (0 , n_channels):
            icp = ic + 1
            channel_names.append( str('Signal-%02d' % icp) )
        
    
    #% Amplitude Modulation Analysis
    # Default Modulation Analysis parameters
    parameters = {}
    parameters['seg_size_sec'] = 8.0      # segment of signal to compute the Modulation Spectrogram (seconds)
    parameters['seg_shft_sec'] = 8.0      # shift between consecutive segments (seconds)
    parameters['win_size_sec'] = 0.5      # window length for the STFFT
    parameters['win_shft_sec'] = 0.02     # shift between consecutive windows (seconds)
    parameters['freq_range']   = None     # limits [min, max] for the conventional frequency axis (Hz)
    parameters['mfreq_range']  = None     # limits [min, max] for the modulation frequency axis (Hz)
    parameters['freq_color']   = None     # limits [min, max] for the power in Spectrogram (dB)
    parameters['mfreq_color']  = None     # limits [min, max] for the power in Modulation Spectrogram (dB)
        
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
     
