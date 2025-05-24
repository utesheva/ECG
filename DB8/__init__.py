"""DB-8 method"""
import spkit as sp


def main(ecg, fs):
    wav = sp.wavelet_filtering(ecg, wv='db8', threshold='optimal',
                               wpd_mode='periodization', WPD=True)
    return wav
