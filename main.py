import pywt
from matplotlib import pyplot as plt
import mne
import scipy.signal as signal
import numpy as np
import pdb
#pdb.set_trace()

NAME = '2.edf'

record = mne.io.read_raw_edf(NAME, preload=True)
#record.plot(duration=10.0)
info = record.info
channels = record.ch_names
print(info)
print(channels)
record_1, times=record.get_data(return_times=True, picks='ECG 1')
print("Record_1\n")
print(record_1)
# REALIZATION WITH .DAT

# Шум у нас реальный,  вносить не надо
#db = 10
#noise = np.random.normal(0, db, len(record))
#record_1 = noise + record


w = pywt.Wavelet('db8')
d_lo, d_hi, r_lo, r_hi = w.filter_bank
print ("R_lo\n",  r_lo)
plt.plot(times, record_1[0], label='Before')
y_filtered = signal.convolve(record_1[0], r_lo, mode='same')

#plt.plot(record, noise, label = 'noise')
#plt.plot(record_1.index, record_1.values, label='before')
plt.plot(times, y_filtered, label='after')
plt.title('db8')
plt.legend()
plt.show()

