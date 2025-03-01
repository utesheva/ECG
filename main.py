import pywt
from matplotlib import pyplot as plt
import mne
import scipy.signal as signal
import numpy as np
import pdb
import spkit as sp

NAME = '2.edf'

record = mne.io.read_raw_edf(NAME, preload=True)
info = record.info
channels = record.ch_names
print(info)
print(channels)
record_1, times=record.get_data(return_times=True, picks='ECG 1')

cA, cD = pywt.dwt(record_1[0], 'db8')
cA1 = pywt.threshold(cA, 1)
cD1 = pywt.threshold(cD, 2)
record_filtered2 = pywt.idwt(cA1, cD1, 'db8')

wav = sp.wavelet_filtering(record_1[0], wv='db8', threshold='optimal',
                           wpd_mode='periodization', WPD=True)
wav2 = sp.wavelet_filtering(wav, wv='db8', threshold='optimal')
plt.plot(times, record_1[0], label = 'Before')
plt.plot(times, wav, label = 'After')
plt.plot(times, wav2, label = 'After 2')
plt.legend()
plt.show()

