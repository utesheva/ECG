import pywt
from matplotlib import pyplot as plt
import mne
import scipy.signal as signal
import numpy as np
import pdb
import spkit as sp
from ser import filtered

NAME = 'Rh10010.edf'

record = mne.io.read_raw_edf(NAME, preload=True)
info = record.info
channels = record.ch_names
print(info)
print(channels)
record_1, times=record.get_data(return_times=True, picks='sig')
'''
wav = sp.wavelet_filtering(record_1[0], wv='db8', threshold='optimal',
                           wpd_mode='periodization', WPD=True)
wav2 = sp.wavelet_filtering(wav, wv='db8', threshold='optimal')
wav3 = sp.wavelet_filtering(record_1[0], wv='db8', threshold='sd')
'''
wav = filtered(record_1[0])
plt.plot(times, record_1[0], label = 'Before')
plt.plot(times, wav, label = 'After')
'''
plt.plot(times, wav2, label = 'After 2')
plt.plot(times, wav3, label = 'After 3')
'''
plt.legend()
plt.show()

