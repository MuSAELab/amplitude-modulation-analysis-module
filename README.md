# Amplitude Modulation Analysis Module

The amplitude modulation analysis module for **Python 3**, provides functions to compute and visualize the frequency-frequency-domain representation of real-valued signals. The **MATLAB-Octave** version of this module can be found here: [https://github.com/MuSAELab/amplitude-modulation-analysis-toolbox](https://github.com/MuSAELab/amplitude-modulation-analysis-toolbox)

The module includes a GUI implementation, which facilitates the amplitude modulation analysis by allowing changing parameters on line.

In summary, the frequency-frequency representation of a signal is computed by performing two transformations.

![diagram](https://user-images.githubusercontent.com/8238803/35760392-c74639f6-084d-11e8-8d34-396324e9b045.png)
Signal processing steps involved in the calculation of the modulation spectrogram from the amplitude spectrogram of signal. The block |abs| indicates the absolute value, and the FT indicates the use of the Fourier transform.

This module provides two implementations for the time to time-frequency transformation (*First Transformation*), one based on the STFFT, and the other on the continuous wavelet transform (CWT) using the Complex Morlet wavelet. The time-frequency to frequency-frequency transformation (*Second Transformation*) is carried out with the FFT.

## Installation
Dowload or clone the respository, then:
`$ pip install .`

## Examples
Besides the functions to compute and visualize the frequency-frequency representation of real signals, example data and scripts are provided.

### Example 1: `example_01.py`
This example shows the usage and the commands accepted by GUI to explore amplitude modulation for a example ECG and EEG data. The GUI can be called with the functions:
`explore_strfft_am_gui()` which uses STFFT, and `explore_wavelet_am_gui()` based on wavelet transformation. Further details in their use refer to the comments in `example_01.py`.  

![stfft](https://user-images.githubusercontent.com/8238803/35760391-c4cee66e-084d-11e8-977d-48f757f72495.png)
STFFT-based Amplitude Modulation analysis GUI  
</br>

![wavelet](https://user-images.githubusercontent.com/8238803/35760382-b1116886-084d-11e8-864e-155ba5359c65.png)
CWT-based Amplitude Modulation analysis GUI

### Example 2: `example_02.py`
This script shows presents the details in the usage of the functions in the module to carry on the signal transformations, as well as plotting functions. Refer to the comments in `example_02.py`

### Acknowledgement
The research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via IARPA Contract N°2017 - 17042800005 . The views and conclusions con - tained herein are thos e of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprint s for Governmental purposes notwithstanding any copyright annotation thereon.
