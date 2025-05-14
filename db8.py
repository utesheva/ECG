"""DB-8 method"""
import pywt
from matplotlib import pyplot as plt
import mne
import numpy as np
import spkit as sp
import sys

if __name__ == '__main__':
    NAME = sys.argv[1]
    RECORD = mne.io.read_raw_edf(NAME, preload=True)
    INFO = RECORD.info
    FS = int(INFO['sfreq'])

    channels = RECORD.ch_names

    record_1, times=RECORD.get_data(return_times=True, picks=channels[0])

def main(ecg, fs):
    wav = sp.wavelet_filtering(ecg, wv='db8', threshold='optimal',
                           wpd_mode='periodization', WPD=True)
    return wav
'''
wav2 = sp.wavelet_filtering(wav, wv='db8', threshold='optimal',
                            wpd_mode='periodization', WPD=True)
'''
#plt.xticks(np.arange(0, 5000, 150))
'''
plt.subplot(2, 1, 1)
plt.plot(record_1[0][:5000], label = 'Исходный сигнал', color='red')
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(wav, label = 'Сигнал после фильтрации', color='blue')
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.legend()
plt.tight_layout()
plt.savefig('db8.png')
'''
