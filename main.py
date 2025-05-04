"""DB-8 method"""
import pywt
from matplotlib import pyplot as plt
import mne
import numpy as np
import spkit as sp
import sys

NAME = sys.argv[1]
RECORD = mne.io.read_raw_edf(NAME, preload=True)
INFO = RECORD.info
FS = int(INFO['sfreq'])

channels = RECORD.ch_names

record_1, times=RECORD.get_data(return_times=True, picks=channels[0])

wav = sp.wavelet_filtering(record_1[0][:5000], wv='db8', threshold='optimal',
                           wpd_mode='periodization', WPD=True)
'''
wav2 = sp.wavelet_filtering(wav, wv='db8', threshold='optimal',
                            wpd_mode='periodization', WPD=True)
'''
plt.xticks(np.arange(0, 5000, 150))
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.plot(record_1[0][:5000], label = 'Исходный сигнал', color='red')
plt.plot(wav, label = 'Сигнал после фильтрации', color='blue')
plt.legend()
plt.show()

