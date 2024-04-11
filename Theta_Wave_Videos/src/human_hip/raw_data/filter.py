#!/usr/bin/env python3  

from scipy import signal 


# Downsample abd remove artifact
def downsample(wav_lfp, dec=20, fs=20000.0):
    wav_data = signal.decimate(wav_lfp, dec)
    return fs/dec, wav_data


# Bandpass filter
# Note: another example of a filter (not used)- https://github.com/hengenlab/neuraltoolkit)
def butter_filter(data, lowcut=None, highcut=None, fs=20000.0, order=5):
    """
    A digital butterworth filter. Type is based on input value.
    Inputs:
        data: array_like data to be filtered
        lowcut: low cutoff frequency. If None or 0, highcut must be a number.
                Filter is lowpass.
        highcut: high cutoff frequency. If None, lowpass must be a non-zero number.
                 Filter is highpass.
        If lowcut and highcut are both give, this filter is bandpass.
        In this case, lowcut must be smaller than highcut.
        fs: sample rate
        order: order of the filter
    Return:
        The filtered output with the same shape as data
    """

    assert (lowcut not in [None, 0]) or (highcut != None), \
        "Need at least a low cutoff (lowcut) or high cutoff (highcut) frequency!"
    if (lowcut != None) and (highcut != None):
        assert lowcut < highcut, "lowcut must be smaller than highcut"

    if lowcut == None or lowcut == 0:
        filter_type = 'lowpass'
        Wn = highcut / fs * 2
    elif highcut == None:
        filter_type = 'highpass'
        Wn = lowcut / fs * 2
    else:
        filter_type = "bandpass"
        band = [lowcut, highcut]
        Wn = [e / fs * 2 for e in band]

    filter_coeff = signal.iirfilter(order, Wn, analog=False, btype=filter_type, output='sos')
    axis = 1 if len(data.shape) == 2 else -1
    filtered_traces = signal.sosfiltfilt(filter_coeff, data, axis=axis)
    return filtered_traces



def get_brain_waves(data, fs=20000):
    waves = { "original": data, "basic": butter_filter( data, lowcut=0.1, highcut=100, fs=fs) }
    waves["low"]   = butter_filter( waves["basic"],  0.1, 0.5, fs=fs)
    waves["delta"] = butter_filter( waves["basic"], 0.5, 4, fs=fs)
    waves["theta"] = butter_filter( waves["basic"], 4, 8, fs=fs)
    waves["alpha"] = butter_filter( waves["basic"], 8, 13, fs=fs)
    waves["beta"]  = butter_filter( waves["basic"], 13, 30, fs=fs)
    waves["gamma"] = butter_filter( waves["basic"], 30, 50, fs=fs)
    return waves
