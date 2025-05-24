"""DB-8 method"""
import pywt
from matplotlib import pyplot as plt
import mne
import numpy as np
import spkit as sp
import sys

def main(ecg, fs):
    wav = sp.wavelet_filtering(ecg, wv='db8', threshold='optimal',
                           wpd_mode='periodization', WPD=True)
    return wav
