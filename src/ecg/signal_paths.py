from scipy.signal import butter, filtfilt
import numpy as np

def display_filter(raw, fs):
    # For plotting + R peak detection
    b, a = butter(4, [0.5/(fs/2), 40/(fs/2)], 'band')
    return filtfilt(b, a, raw)

def measurement_filter(raw, fs):
    # For PR, QRS, QT, QTc (clinical)
    b, a = butter(4, [0.05/(fs/2), min(150/(fs/2), 0.99)], 'band')
    return filtfilt(b, a, raw)
